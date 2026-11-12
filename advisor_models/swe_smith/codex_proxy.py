"""Proxy between Codex and OpenRouter for the E7 Codex harness pilot.

Codex sends its Responses API requests here instead of to OpenRouter. The
proxy counts them (one request = one agent step), refuses the request past
the step cap, and, in the advisor arm, appends advice to the request input
the way mini-swe-agent's `add_advice` appends it to the last message.

Codex resends its whole history on every request, so advice added to one
request would be gone from the next. The proxy remembers which input item
each advice went on and re-applies it on every later request, so the agent
sees the same history it would in mini-swe-agent. Advice whose item is no
longer in the input (Codex compacted its history) is counted as lost.

See experiments/e07-route-then-advise/spec.md in hecate, "Exploratory Codex
harness pilot".
"""

from __future__ import annotations

import copy
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional

import requests

# Upstream retries. OpenRouter sometimes routes a request to a Google key that
# answers 404 for gemini-2.5-flash-lite ("no longer available to new users");
# the next attempt usually goes elsewhere.
UPSTREAM_ATTEMPTS = 8
RETRY_STATUSES = {404, 408, 409, 429, 500, 502, 503, 504, 529}
RETRY_BACKOFF_S = 2.0
RETRY_BACKOFF_MAX_S = 30.0

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# mini-swe-agent's DefaultAgent.add_advice wrapper, verbatim.
ADVICE_WRAPPER = """<advice>
An advisor specializing in software engineering has suggested the following advice:
{advice}
Consider utilizing this advice as you continue.
</advice>"""

OUTPUT_TYPES = {
    "function_call_output",
    "custom_tool_call_output",
    "local_shell_call_output",
}


def format_advice(advice: str) -> str:
    return ADVICE_WRAPPER.format(advice=advice)


def _parts_text(content: Any) -> str:
    """Text of a message content or tool output: a string or a list of parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
        return "\n".join(texts)
    return json.dumps(content)


def item_key(item: Dict[str, Any], index: int) -> str:
    """Stable id of an input item across requests.

    Tool outputs carry a call_id that Codex keeps when it resends history.
    Messages have none, so they are keyed by position, which holds while
    history only grows (it does not across a compaction; see module doc).
    """
    item_type = item.get("type", "message")
    if item_type in OUTPUT_TYPES and item.get("call_id"):
        return f"{item_type}:{item['call_id']}"
    return f"{item_type}@{index}"


def history_to_messages(items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Map Responses API input items to mini-swe-agent style messages.

    Assistant text and tool calls become assistant messages, tool outputs
    and user text become user messages. Reasoning items are left out; they
    are encrypted.
    """
    messages = []
    for item in items:
        item_type = item.get("type", "message")
        if item_type == "message":
            role = item.get("role", "user")
            if role == "developer":
                role = "system"
            messages.append({"role": role, "content": _parts_text(item.get("content"))})
        elif item_type == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": f"{item.get('name', '')}({item.get('arguments', '')})",
                }
            )
        elif item_type == "custom_tool_call":
            messages.append(
                {"role": "assistant", "content": f"{item.get('name', '')}:\n{item.get('input', '')}"}
            )
        elif item_type == "local_shell_call":
            action = item.get("action") or {}
            messages.append({"role": "assistant", "content": json.dumps(action.get("command", action))})
        elif item_type in OUTPUT_TYPES:
            messages.append({"role": "user", "content": _parts_text(item.get("output"))})
    return messages


def append_text(item: Dict[str, Any], text: str) -> bool:
    """Append text to an item in place. False if the item has no text to extend."""
    item_type = item.get("type", "message")
    field = "output" if item_type in OUTPUT_TYPES else "content"
    if item_type not in OUTPUT_TYPES and item_type != "message":
        return False
    value = item.get(field)
    if isinstance(value, str):
        item[field] = value + "\n" + text
        return True
    if isinstance(value, list):
        part_type = "output_text" if item.get("role") == "assistant" else "input_text"
        value.append({"type": part_type, "text": "\n" + text})
        return True
    return False


class RunState:
    """Per-run counters and advice, shared by the proxy's request threads."""

    def __init__(
        self,
        step_limit: int = 40,
        advisor: Optional[Callable[[int, List[Dict[str, Any]]], Optional[str]]] = None,
        advice_interval: int = 5,
        log_path: Optional[str] = None,
        drop_reasoning: bool = False,
        first_request_path: Optional[str] = None,
    ):
        self.step_limit = step_limit
        self.advisor = advisor
        self.advice_interval = advice_interval
        self.log_path = log_path
        # Strip reasoning settings before OpenRouter, so a student that ran
        # without thinking in E7 (Flash-Lite) runs without it here too.
        self.drop_reasoning = drop_reasoning
        self.first_request_path = first_request_path
        self.lock = threading.Lock()
        self.n_requests = 0
        self.cap_hit = False
        # (key, wrapped advice) in delivery order
        self.advice: List[tuple[str, str]] = []
        self.advisor_calls = 0
        self.advice_lost = 0
        self.fallback_inserts = 0
        self.tool_calls = 0
        self.upstream_errors = 0
        # Requests still failing after every retry; each one reached Codex.
        self.failed_requests = 0
        self.stateful_requests = 0
        self.usage = {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
        self.cost = 0.0

    def log(self, entry: Dict[str, Any]):
        if not self.log_path:
            return
        entry = {"t": time.time(), **entry}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def advice_due(self, n: int) -> bool:
        """Initial advice before request 1, then before requests 6, 11, ..."""
        return self.advisor is not None and (n - 1) % self.advice_interval == 0

    def apply_advice(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Re-apply every past advice to its item. Returns a new list."""
        items = copy.deepcopy(items)
        keys = {item_key(item, i): i for i, item in enumerate(items)}
        inserts = []
        lost = 0
        for key, text in self.advice:
            if key not in keys:
                lost += 1
                continue
            idx = keys[key]
            if not append_text(items[idx], text):
                inserts.append((idx, text))
        for idx, text in sorted(inserts, reverse=True):
            items.insert(
                idx + 1,
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]},
            )
        return items, lost, len(inserts)

    def prepare(self, body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Count one step and return the body to forward, or None past the cap."""
        with self.lock:
            self.n_requests += 1
            n = self.n_requests
            if n > self.step_limit:
                self.cap_hit = True
                self.log({"event": "cap", "request": n})
                return None
            if body.get("previous_response_id"):
                # History is then server-side and advice cannot be re-applied.
                self.stateful_requests += 1
            items = body.get("input")
            if isinstance(items, str):
                items = [{"type": "message", "role": "user", "content": items}]
            items = items or []
            if self.advice_due(n) and items:
                with_past, _, _ = self.apply_advice(items)
                guidance = self.advisor(n, with_past)
                self.advisor_calls += 1
                if guidance:
                    key = item_key(items[-1], len(items) - 1)
                    self.advice.append((key, format_advice(guidance)))
                    self.log({"event": "advice", "request": n, "key": key, "advice": guidance})
            new_items, lost, inserts = self.apply_advice(items)
            self.advice_lost = max(self.advice_lost, lost)
            self.fallback_inserts += inserts
            out = dict(body)
            out["input"] = new_items
            if self.drop_reasoning:
                out.pop("reasoning", None)
                include = [x for x in out.get("include") or [] if not x.startswith("reasoning")]
                if include:
                    out["include"] = include
                else:
                    out.pop("include", None)
            if n == 1 and self.first_request_path:
                with open(self.first_request_path, "w") as f:
                    json.dump(out, f, indent=2)
            if n == 1:
                # What Codex sends, once per run: the record of its settings.
                self.log({
                    "event": "settings",
                    "model": out.get("model"),
                    "reasoning": out.get("reasoning"),
                    "include": out.get("include"),
                    "tools": [t.get("name") or t.get("type") for t in out.get("tools") or []],
                    "instructions_chars": len(out.get("instructions") or ""),
                })
            self.log({"event": "request", "request": n, "n_items": len(items), "advice_applied": len(self.advice) - lost, "advice_lost": lost})
            return out

    def record_response(self, response: Dict[str, Any], prices: Dict[str, float]):
        """Add usage and tool calls of one completed response."""
        usage = response.get("usage") or {}
        details = usage.get("input_tokens_details") or {}
        out_details = usage.get("output_tokens_details") or {}
        inp = usage.get("input_tokens", 0) or 0
        cached = details.get("cached_tokens", 0) or 0
        out = usage.get("output_tokens", 0) or 0
        self.log({"event": "response", "id": response.get("id"), "usage": usage})
        with self.lock:
            self.usage["input_tokens"] += inp
            self.usage["cached_tokens"] += cached
            self.usage["output_tokens"] += out
            self.usage["reasoning_tokens"] += out_details.get("reasoning_tokens", 0) or 0
            if usage.get("cost") is not None:
                self.cost += float(usage["cost"])
            else:
                self.cost += (
                    (inp - cached) * prices.get("input", 0.0)
                    + cached * prices.get("cached_input", 0.0)
                    + out * prices.get("output", 0.0)
                ) / 1e6
            for item in response.get("output") or []:
                if item.get("type") in ("function_call", "custom_tool_call", "local_shell_call"):
                    self.tool_calls += 1

    def summary(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "steps": min(self.n_requests, self.step_limit),
                "cap_hit": self.cap_hit,
                "advisor_calls": self.advisor_calls,
                "advice_delivered": len(self.advice),
                "advice_lost": self.advice_lost,
                "fallback_inserts": self.fallback_inserts,
                "tool_calls": self.tool_calls,
                "upstream_errors": self.upstream_errors,
                "failed_requests": self.failed_requests,
                "stateful_requests": self.stateful_requests,
                "usage": dict(self.usage),
                "cost": self.cost,
            }


def _completed_response(sse_text: str) -> Optional[Dict[str, Any]]:
    """The response object of the last response.completed event in an SSE body."""
    found = None
    for line in sse_text.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if not data or data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "response.completed":
            found = event.get("response")
    return found


def make_handler(
    state: RunState,
    api_key: str,
    upstream: str,
    prices: Dict[str, float],
    upstream_model: Optional[str] = None,
):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _upstream_url(self) -> str:
            path = self.path
            if path.startswith("/v1"):
                path = path[len("/v1"):]
            return upstream.rstrip("/") + path

        def _send_json(self, status: int, payload: Dict[str, Any]):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            resp = requests.get(
                self._upstream_url(), headers={"Authorization": f"Bearer {api_key}"}, timeout=60
            )
            self.send_response(resp.status_code)
            self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(resp.content)))
            self.end_headers()
            self.wfile.write(resp.content)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            if not self.path.rstrip("/").endswith("/responses"):
                self._send_json(404, {"error": {"message": f"proxy: unsupported path {self.path}"}})
                return
            body = json.loads(raw)
            if upstream_model:
                body["model"] = upstream_model
            out = state.prepare(body)
            if out is None:
                self._send_json(
                    400, {"error": {"message": "step cap reached", "type": "invalid_request_error"}}
                )
                return
            # Provider errors are retried here, with backoff, so they do not
            # reach Codex (which retries at once and then ends the run). One
            # Codex request is one step however many attempts it takes.
            resp = None
            for attempt in range(1, UPSTREAM_ATTEMPTS + 1):
                try:
                    resp = requests.post(
                        self._upstream_url(),
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                            "Accept": self.headers.get("Accept", "text/event-stream"),
                        },
                        data=json.dumps(out),
                        stream=True,
                        timeout=600,
                    )
                except requests.RequestException as e:
                    resp = None
                    status, detail = None, repr(e)[:2000]
                else:
                    if resp.status_code < 400:
                        break
                    status, detail = resp.status_code, resp.text[:2000]
                with state.lock:
                    state.upstream_errors += 1
                state.log({"event": "upstream_error", "status": status, "attempt": attempt, "body": detail})
                if status is not None and status not in RETRY_STATUSES:
                    break
                if attempt < UPSTREAM_ATTEMPTS:
                    time.sleep(min(RETRY_BACKOFF_S * 2 ** (attempt - 1), RETRY_BACKOFF_MAX_S))
            if resp is None:
                with state.lock:
                    state.failed_requests += 1
                self._send_json(502, {"error": {"message": f"proxy: upstream unreachable: {detail}"}})
                return
            if resp.status_code >= 400:
                with state.lock:
                    state.failed_requests += 1
            self.send_response(resp.status_code)
            self.send_header("Content-Type", resp.headers.get("Content-Type", "text/event-stream"))
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            # Codex hangs up as soon as it has response.completed. Keep reading
            # upstream after that so the usage is still recorded.
            client_open = True
            chunks = []
            for chunk in resp.iter_content(chunk_size=None):
                if not chunk:
                    continue
                chunks.append(chunk)
                if client_open:
                    try:
                        self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        client_open = False
            if client_open:
                try:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            self.close_connection = True
            if resp.status_code < 400:
                text = b"".join(chunks).decode("utf-8", errors="replace")
                if "text/event-stream" in resp.headers.get("Content-Type", ""):
                    completed = _completed_response(text)
                else:
                    try:
                        completed = json.loads(text)
                    except json.JSONDecodeError:
                        completed = None
                if completed:
                    state.record_response(completed, prices)

    return Handler


class CodexProxy:
    """One proxy per run, on a free local port, in a background thread.

    upstream_model replaces the request's model, so Codex can run under a
    name it has metadata for (gpt-5.4-nano) while OpenRouter gets its own id.
    """

    def __init__(
        self,
        state: RunState,
        api_key: str,
        upstream: str = OPENROUTER_BASE,
        prices: Optional[Dict[str, float]] = None,
        upstream_model: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 0,
    ):
        self.state = state
        self.server = ThreadingHTTPServer(
            (host, port), make_handler(state, api_key, upstream, prices or {}, upstream_model)
        )
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
