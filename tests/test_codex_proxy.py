"""Offline tests of the E7 Codex proxy. No provider call: a fake upstream.

Run: .venv/bin/python -m unittest tests/test_codex_proxy.py
"""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

from advisor_models.swe_smith import codex_proxy
from advisor_models.swe_smith.codex_proxy import (
    CodexProxy,
    RunState,
    format_advice,
    history_to_messages,
)


class FakeUpstream:
    """Records request bodies and answers with one SSE response.completed."""

    def __init__(self, fail_first=0, fail_status=404):
        self.bodies = []
        self.attempts = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                outer.attempts += 1
                if outer.attempts <= fail_first:
                    err = json.dumps({"error": {"message": "Provider returned error", "code": fail_status}}).encode()
                    self.send_response(fail_status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(err)))
                    self.end_headers()
                    self.wfile.write(err)
                    return
                outer.bodies.append(json.loads(body))
                completed = {
                    "type": "response.completed",
                    "response": {
                        "output": [{"type": "function_call", "name": "shell", "arguments": "{}", "call_id": "x"}],
                        "usage": {
                            "input_tokens": 1000,
                            "input_tokens_details": {"cached_tokens": 600},
                            "output_tokens": 50,
                            "output_tokens_details": {"reasoning_tokens": 30},
                        },
                    },
                }
                data = f"event: response.completed\ndata: {json.dumps(completed)}\n\n".encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def task_message():
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Fix the bug."}]}


def turn(i):
    """One Codex turn: a tool call and its output."""
    return [
        {"type": "function_call", "name": "shell", "arguments": json.dumps({"command": ["ls", str(i)]}), "call_id": f"c{i}"},
        {"type": "function_call_output", "call_id": f"c{i}", "output": f"out {i}"},
    ]


class ProxyTest(unittest.TestCase):
    def setUp(self):
        self.upstream = FakeUpstream()

    def tearDown(self):
        self.upstream.close()

    def post(self, proxy, items):
        return requests.post(
            f"http://127.0.0.1:{proxy.port}/v1/responses",
            json={"model": "m", "input": items, "stream": True},
            timeout=10,
        )

    def test_pass_through_is_unchanged_and_counted(self):
        state = RunState(step_limit=40)
        items = [task_message()]
        with CodexProxy(state, "k", upstream=self.upstream.url, prices={"input": 0.2, "cached_input": 0.02, "output": 1.25}) as proxy:
            for i in range(3):
                resp = self.post(proxy, items)
                self.assertEqual(resp.status_code, 200)
                self.assertIn("response.completed", resp.text)
                items = items + turn(i)
        sent = [b["input"] for b in self.upstream.bodies]
        self.assertEqual(sent[0], [task_message()])
        self.assertEqual(sent[2], [task_message()] + turn(0) + turn(1))
        s = state.summary()
        self.assertEqual(s["steps"], 3)
        self.assertEqual(s["tool_calls"], 3)
        self.assertEqual(s["usage"]["input_tokens"], 3000)
        self.assertAlmostEqual(s["cost"], 3 * (400 * 0.2 + 600 * 0.02 + 50 * 1.25) / 1e6)

    def test_upstream_model_replaces_codex_model(self):
        state = RunState(step_limit=40)
        with CodexProxy(state, "k", upstream=self.upstream.url, upstream_model="openai/gpt-5.4-nano") as proxy:
            self.post(proxy, [task_message()])
        self.assertEqual(self.upstream.bodies[0]["model"], "openai/gpt-5.4-nano")

    def test_drop_reasoning_strips_reasoning_settings(self):
        state = RunState(step_limit=40, drop_reasoning=True)
        with CodexProxy(state, "k", upstream=self.upstream.url) as proxy:
            requests.post(
                f"http://127.0.0.1:{proxy.port}/v1/responses",
                json={"model": "m", "input": [task_message()], "reasoning": {"effort": "medium"},
                      "include": ["reasoning.encrypted_content"]},
                timeout=10,
            )
        sent = self.upstream.bodies[0]
        self.assertNotIn("reasoning", sent)
        self.assertNotIn("include", sent)

    def test_cap_refuses_request_past_limit(self):
        state = RunState(step_limit=2)
        with CodexProxy(state, "k", upstream=self.upstream.url) as proxy:
            self.assertEqual(self.post(proxy, [task_message()]).status_code, 200)
            self.assertEqual(self.post(proxy, [task_message()]).status_code, 200)
            self.assertEqual(self.post(proxy, [task_message()]).status_code, 400)
        self.assertTrue(state.cap_hit)
        self.assertEqual(len(self.upstream.bodies), 2)
        self.assertEqual(state.summary()["steps"], 2)

    def test_advice_timing_placement_and_reapplication(self):
        calls = []

        def advisor(n, items):
            calls.append((n, items))
            return f"advice {n}"

        state = RunState(step_limit=40, advisor=advisor)
        items = [task_message()]
        with CodexProxy(state, "k", upstream=self.upstream.url) as proxy:
            for i in range(7):
                self.post(proxy, items)
                items = items + turn(i)
        # Initial advice before request 1, follow-up before request 6.
        self.assertEqual([n for n, _ in calls], [1, 6])
        first = self.upstream.bodies[0]["input"]
        self.assertEqual(first[0]["content"][-1]["text"], "\n" + format_advice("advice 1"))
        # Request 6: last item is the output of turn 4; advice 6 goes on it.
        sixth = self.upstream.bodies[5]["input"]
        self.assertTrue(sixth[-1]["output"].endswith(format_advice("advice 6")))
        # Request 7 still carries both advices in the same places.
        seventh = self.upstream.bodies[6]["input"]
        self.assertEqual(seventh[0]["content"][-1]["text"], "\n" + format_advice("advice 1"))
        self.assertTrue(seventh[10]["output"].endswith(format_advice("advice 6")))
        self.assertFalse(seventh[-1]["output"].endswith("</advice>"))
        # The advisor's follow-up input includes the earlier advice, as in mini-swe-agent.
        _, seen = calls[1]
        self.assertIn("advice 1", history_to_messages(seen)[0]["content"])
        # Codex's own copy is never modified.
        self.assertEqual(items[0], task_message())
        self.assertEqual(state.summary()["advice_delivered"], 2)

    def test_advice_lost_after_compaction(self):
        state = RunState(step_limit=40, advisor=lambda n, items: f"advice {n}")
        items = [task_message()]
        with CodexProxy(state, "k", upstream=self.upstream.url) as proxy:
            for i in range(6):
                self.post(proxy, items)
                items = items + turn(i)
            # Codex replaces history with a summary message; advice 6 was on c4's output.
            compacted = [task_message(), {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "summary"}]}]
            self.post(proxy, compacted)
        self.assertEqual(state.summary()["advice_lost"], 1)

    def test_history_to_messages(self):
        items = [
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "rules"}]},
            task_message(),
            {"type": "reasoning", "encrypted_content": "zzz"},
            {"type": "custom_tool_call", "name": "apply_patch", "input": "*** Begin Patch", "call_id": "p"},
            {"type": "custom_tool_call_output", "call_id": "p", "output": "Done"},
        ]
        self.assertEqual(
            history_to_messages(items),
            [
                {"role": "system", "content": "rules"},
                {"role": "user", "content": "Fix the bug."},
                {"role": "assistant", "content": "apply_patch:\n*** Begin Patch"},
                {"role": "user", "content": "Done"},
            ],
        )


class RetryTest(unittest.TestCase):
    def setUp(self):
        self._backoff = codex_proxy.RETRY_BACKOFF_S
        codex_proxy.RETRY_BACKOFF_S = 0.0

    def tearDown(self):
        codex_proxy.RETRY_BACKOFF_S = self._backoff

    def post(self, proxy):
        return requests.post(
            f"http://127.0.0.1:{proxy.port}/v1/responses",
            json={"model": "m", "input": [task_message()], "stream": True},
            timeout=10,
        )

    def test_provider_404_is_retried_and_counted_once(self):
        upstream = FakeUpstream(fail_first=3)
        try:
            state = RunState(step_limit=40)
            with CodexProxy(state, "k", upstream=upstream.url) as proxy:
                resp = self.post(proxy)
            self.assertEqual(resp.status_code, 200)
            s = state.summary()
            self.assertEqual(s["steps"], 1)
            self.assertEqual(s["upstream_errors"], 3)
            self.assertEqual(s["failed_requests"], 0)
            self.assertEqual(upstream.attempts, 4)
        finally:
            upstream.close()

    def test_failure_after_all_attempts_reaches_codex(self):
        upstream = FakeUpstream(fail_first=100)
        try:
            state = RunState(step_limit=40)
            with CodexProxy(state, "k", upstream=upstream.url) as proxy:
                resp = self.post(proxy)
            self.assertEqual(resp.status_code, 404)
            self.assertEqual(upstream.attempts, codex_proxy.UPSTREAM_ATTEMPTS)
            self.assertEqual(state.summary()["failed_requests"], 1)
        finally:
            upstream.close()

    def test_non_retryable_status_is_not_retried(self):
        upstream = FakeUpstream(fail_first=100, fail_status=400)
        try:
            state = RunState(step_limit=40)
            with CodexProxy(state, "k", upstream=upstream.url) as proxy:
                resp = self.post(proxy)
            self.assertEqual(resp.status_code, 400)
            self.assertEqual(upstream.attempts, 1)
        finally:
            upstream.close()


if __name__ == "__main__":
    unittest.main()
