"""E10 on the DGX Spark: the policy's forward pass without full logits or padding.

G1 attempt 7 (16 Nov, spark/README.md in hecate) ran all 32 rollouts and was
then killed within ~20 s of the trainer's first forward pass, with about 8 GiB
of the GB10's memory free. Upstream SkyRL's FSDP forward
(``HFModelWrapper.forward``, skyrl/backends/skyrl_train/workers/model_wrapper.py
at 3e8700f) does two things that do not fit in that:

  1. It builds the full logits ``[B, S, 152064]`` and only then takes the
     log-probs (its 1,024-token chunking is for the entropy, applied to the
     already-built logits). At S = 26,798 that is 8.15 GB in bf16 per
     sequence, before anything else in the pass.
  2. Every micro-batch (1 sequence here) is padded to the batch's longest
     sequence, and with flash-attn unavailable on the GB10
     (``remove_microbatch_padding=false``) the padding is computed and an
     explicit attention mask is passed, which keeps SDPA off its causal
     fast path. A 3,000-token rollout costs as much as the 26,798-token one.

``lean_forward`` (installed in each policy worker by ``install_lean_forward``)
computes the same per-token log-probs and entropy without either:

  - each sequence's valid (attention-mask 1) columns are run on their own,
    with upstream's position ids and no mask, so attention is causal SDPA over
    exactly the tokens upstream attends to; outputs are scattered back to the
    padded layout (padding columns get 0, where upstream gives values that
    every consumer masks out);
  - the LM head is applied in ``logprobs_chunk_size``-token chunks inside an
    autograd Function that recomputes each chunk's logits in backward, so no
    ``[S, vocab]`` tensor is ever held. Labels are upstream's (the sequence
    rolled by one), temperature is applied to each chunk's logits as upstream
    applies ``div_`` to the full tensor, and the log-prob is computed in fp32
    as flash-attn's cross-entropy does; the entropy follows
    ``chunked_entropy_from_logits`` (bf16 log_softmax per chunk).
  - SDPA's math backend (which builds the full L x L attention matrix, ~40 GB
    at 26.8k tokens for 28 heads) is excluded during the forward on CUDA, so
    a shape no efficient kernel supports fails loudly instead of using it.

Nothing the spec fixes changes: same tokens, lengths, batch, LoRA, KL and
temperature; only which intermediate tensors exist. Configurations this does
not cover (VLM inputs, sequence parallelism, sample packing, sample-support
replay, entropy with gradient) fall through to upstream's forward unchanged.
Set ``E10_LEAN_FORWARD=0`` to use upstream's forward everywhere.
"""

import contextlib
import os
import threading
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

_request = threading.local()
_ORIGINAL_FORWARD = None


class ChunkedLMHeadLogprob(torch.autograd.Function):
    """Log-prob of each label (and optionally entropy) from hidden states, the LM head
    applied chunk by chunk; backward recomputes each chunk's logits."""

    @staticmethod
    def forward(ctx, hidden, weight, labels, temperature, chunk_size, compute_entropy):
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            n = hidden.shape[0]
            # fp32 for bf16 inputs, as flash-attn's cross-entropy; wider inputs keep their precision
            acc = torch.promote_types(hidden.dtype, torch.float32)
            logp = torch.empty(n, dtype=acc, device=hidden.device)
            ent = torch.zeros(n if compute_entropy else 0, dtype=hidden.dtype, device=hidden.device)
            for s in range(0, n, chunk_size):
                e = min(n, s + chunk_size)
                logits = F.linear(hidden[s:e], weight)
                if temperature != 1.0:
                    logits = logits / temperature
                lf = logits.to(acc)
                logp[s:e] = lf.gather(-1, labels[s:e, None]).squeeze(-1) - torch.logsumexp(lf, dim=-1)
                if compute_entropy:
                    lp = F.log_softmax(logits, dim=-1)
                    ent[s:e] = -(lp.exp() * lp).sum(-1)
                del logits, lf
        ctx.save_for_backward(hidden, weight, labels)
        ctx.temperature = temperature
        ctx.chunk_size = chunk_size
        ctx.mark_non_differentiable(ent)
        return logp, ent

    @staticmethod
    def backward(ctx, grad_logp, grad_ent):
        hidden, weight, labels = ctx.saved_tensors
        need_h, need_w = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        t = ctx.temperature
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            acc = torch.promote_types(hidden.dtype, torch.float32)
            grad_h = torch.zeros_like(hidden) if need_h else None
            grad_w = torch.zeros(weight.shape, dtype=acc, device=weight.device) if need_w else None
            for s in range(0, hidden.shape[0], ctx.chunk_size):
                e = min(hidden.shape[0], s + ctx.chunk_size)
                logits = F.linear(hidden[s:e], weight)
                if t != 1.0:
                    logits = logits / t
                g = grad_logp[s:e].to(acc)[:, None]
                # d logp_label / d logits = onehot(label) - softmax
                grad_logits = torch.softmax(logits.to(acc), dim=-1).mul_(-g)
                grad_logits.scatter_add_(-1, labels[s:e, None], g)
                if t != 1.0:
                    grad_logits.div_(t)
                if need_h:
                    grad_h[s:e] = grad_logits.to(weight.dtype) @ weight
                if need_w:
                    grad_w += grad_logits.t() @ hidden[s:e].to(acc)
                del logits, grad_logits
        return grad_h, (grad_w.to(weight.dtype) if need_w else None), None, None, None, None


def _causal_lm(model: torch.nn.Module) -> torch.nn.Module:
    """The module with ``get_output_embeddings`` under PeftModel / FSDP wrappers."""
    for candidate in (model, getattr(model, "base_model", None),
                      getattr(getattr(model, "base_model", None), "model", None)):
        if candidate is not None and hasattr(candidate, "get_output_embeddings") \
                and candidate.get_output_embeddings() is not None:
            return candidate
    raise RuntimeError("E10 lean forward: no LM head found on the policy model")


def _lm_head_forward(self, hidden_states, *args, **kwargs):
    """The LM head, replaced in place: with a lean-forward request active it returns
    [..., 2] = (label log-prob, entropy) instead of logits. Running inside the
    model's own forward keeps FSDP's parameter handling exactly as for logits."""
    req = getattr(_request, "value", None)
    if req is None:
        return self._e10_original_forward(hidden_states, *args, **kwargs)
    req["logits_dtype"] = self.weight.dtype
    lead = hidden_states.shape[:-1]
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    if self.bias is not None:
        raise RuntimeError("E10 lean forward: LM head with a bias is not supported")
    logp, ent = ChunkedLMHeadLogprob.apply(flat, self.weight, req["labels"].reshape(-1),
                                           req["temperature"], req["chunk_size"], req["compute_entropy"])
    ent = ent.float() if req["compute_entropy"] else torch.zeros_like(logp)
    return torch.stack([logp, ent], dim=-1).reshape(*lead, 2)


def _patch_lm_head(wrapper) -> None:
    head = _causal_lm(wrapper.model).get_output_embeddings()
    if getattr(head, "_e10_lean", False):
        return
    head._e10_original_forward = head.forward
    head.forward = _lm_head_forward.__get__(head)
    head._e10_lean = True


@contextlib.contextmanager
def _active(req):
    _request.value = req
    try:
        yield
    finally:
        _request.value = None


def _no_math_sdpa(device: torch.device):
    if device.type != "cuda":
        return contextlib.nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION])


def lean_forward(self, sequences, num_actions, attention_mask=None, temperature=1.0, return_output=False,
                 compute_entropy=False, entropy_requires_grad=True, pixel_values=None, image_grid_thw=None,
                 mm_token_type_ids=None, sample_support=None, loss_mask=None,
                 enable_sample_support_replay=False, centering_weight=None):
    """HFModelWrapper.forward with the same outputs, minus full logits and padding."""
    if (attention_mask is None or self.is_vlm or self.remove_microbatch_padding
            or self.sequence_parallel_size > 1 or enable_sample_support_replay or centering_weight is not None
            or pixel_values is not None or image_grid_thw is not None
            or (compute_entropy and entropy_requires_grad)):
        return _ORIGINAL_FORWARD(self, sequences, num_actions, attention_mask=attention_mask,
                                 temperature=temperature, return_output=return_output,
                                 compute_entropy=compute_entropy, entropy_requires_grad=entropy_requires_grad,
                                 pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                                 mm_token_type_ids=mm_token_type_ids, sample_support=sample_support,
                                 loss_mask=loss_mask, enable_sample_support_replay=enable_sample_support_replay,
                                 centering_weight=centering_weight)
    _patch_lm_head(self)
    batch, seqlen = attention_mask.shape
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    labels = torch.roll(sequences, shifts=-1, dims=1)

    log_probs = torch.zeros((batch, seqlen), dtype=torch.float32, device=sequences.device)
    entropy = torch.zeros((batch, seqlen), dtype=torch.float32, device=sequences.device) if compute_entropy else None
    rows_lp, rows_ent, rows_idx = [], [], []
    logits_dtype = torch.float32
    for b in range(batch):
        idx = attention_mask[b].nonzero(as_tuple=True)[0]
        req = dict(labels=labels[b, idx][None], temperature=temperature,
                   chunk_size=self.logprobs_chunk_size, compute_entropy=compute_entropy)
        with _active(req), _no_math_sdpa(sequences.device):
            out = self.model(sequences[b, idx][None], attention_mask=None, position_ids=position_ids[b, idx][None])
        logits_dtype = req.get("logits_dtype", logits_dtype)
        packed = out["logits"][0]
        rows_idx.append(idx)
        rows_lp.append(packed[:, 0])
        rows_ent.append(packed[:, 1])
    # Scatter back to the padded layout; index_put keeps the graph to each row's log-probs.
    for b, idx in enumerate(rows_idx):
        log_probs = log_probs.index_put((torch.full_like(idx, b), idx), rows_lp[b])
        if compute_entropy:
            entropy = entropy.index_put((torch.full_like(idx, b), idx), rows_ent[b].detach())

    if isinstance(num_actions, list):
        if len(num_actions) != 1:
            raise RuntimeError("E10 lean forward: per-row num_actions needs sample packing (not used in E10)")
        num_actions = num_actions[0]
    action_log_probs = log_probs[:, -num_actions - 1: -1]
    if not return_output:
        return action_log_probs
    output: Dict[str, Any] = {"logits": None}
    if compute_entropy:
        # upstream's entropy has the logits' dtype: the LM head's (bf16 on the Spark)
        output["entropy"] = entropy.to(logits_dtype)
    return action_log_probs, output


def install_lean_forward() -> bool:
    """Replace HFModelWrapper.forward in this process (a policy worker). Idempotent."""
    global _ORIGINAL_FORWARD
    if os.environ.get("E10_LEAN_FORWARD", "1") == "0":
        return False
    from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper
    if getattr(HFModelWrapper.forward, "_e10_lean", False):
        return True
    _ORIGINAL_FORWARD = HFModelWrapper.forward
    lean_forward._e10_lean = True
    HFModelWrapper.forward = lean_forward
    return True
