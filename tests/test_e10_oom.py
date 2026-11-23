"""Offline tests of E10's lean policy forward (spark_oom.py): same log-probs, entropy
and gradients as upstream SkyRL's HFModelWrapper.forward, on a tiny random Qwen2.

Needs torch, transformers, peft and upstream SkyRL, so it runs in the trainer
image on the CPU (no GPU, no network, no weights):

    docker run --rm -v "$PWD":/w:ro -w /w --entrypoint python \
        hecate/skyrl-upstream-e10:3e8700f -m unittest tests/test_e10_oom.py

Elsewhere every test is skipped.
"""

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HAVE_STACK = all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers", "peft", "skyrl"))

if HAVE_STACK:
    import torch
    import torch.nn.functional as F
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from advisor_models.swe_smith import spark_oom
    from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper


def tiny_model(lora: bool):
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                      tie_word_embeddings=False, attn_implementation="sdpa")
    model = Qwen2ForCausalLM(cfg).float()
    if lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=4, lora_alpha=4, target_modules="all-linear", lora_dropout=0.0))
        with torch.no_grad():  # non-zero LoRA B, so the adapter changes the output
            for name, p in model.named_parameters():
                if "lora_B" in name:
                    p.normal_(0, 0.05)
    return model


def padded_batch():
    """Two rows: left-padded prompts, right-padded responses, as SkyRL batches them."""
    torch.manual_seed(1)
    seq = torch.randint(3, 97, (2, 20))
    mask = torch.ones(2, 20, dtype=torch.long)
    mask[0, :3] = 0          # left padding of row 0's prompt
    mask[1, :1] = 0
    mask[0, 17:] = 0         # right padding of row 0's response
    seq[mask == 0] = 0
    num_actions = 8          # response = last 8 columns
    return seq, mask, num_actions


def wrapper(model, chunk):
    w = HFModelWrapper(model, use_flash_attention_2=False, bf16=False, logprobs_chunk_size=chunk)
    w.eval()
    return w


@unittest.skipUnless(HAVE_STACK, "needs torch, transformers, peft and SkyRL (run in the trainer image)")
class ChunkedHeadTest(unittest.TestCase):
    def test_values_match_log_softmax(self):
        torch.manual_seed(0)
        h, w = torch.randn(13, 8, dtype=torch.float64), torch.randn(11, 8, dtype=torch.float64)
        labels = torch.randint(0, 11, (13,))
        for t in (1.0, 0.7):
            lp, ent = spark_oom.ChunkedLMHeadLogprob.apply(h, w, labels, t, 4, True)
            ref = F.log_softmax(F.linear(h, w) / t, -1)
            torch.testing.assert_close(lp.double(), ref.gather(-1, labels[:, None]).squeeze(-1))
            torch.testing.assert_close(ent, -(ref.exp() * ref).sum(-1))

    def test_gradcheck(self):
        torch.manual_seed(0)
        h = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
        w = torch.randn(6, 5, dtype=torch.float64, requires_grad=True)
        labels = torch.randint(0, 6, (7,))
        f = lambda h, w: spark_oom.ChunkedLMHeadLogprob.apply(h, w, labels, 0.8, 3, False)[0]  # noqa: E731
        self.assertTrue(torch.autograd.gradcheck(f, (h, w)))


@unittest.skipUnless(HAVE_STACK, "needs torch, transformers, peft and SkyRL (run in the trainer image)")
class LeanForwardTest(unittest.TestCase):
    def setUp(self):
        self.original = HFModelWrapper.forward
        spark_oom._ORIGINAL_FORWARD = self.original

    def tearDown(self):
        HFModelWrapper.forward = self.original

    def compare(self, lora, compute_entropy, temperature=1.0, grads=True, checkpointing=False,
                dtype=None, atol=1e-5, rtol=1e-4):
        seq, mask, n = padded_batch()
        action_mask = mask[:, -n:].float()
        results = []
        for lean in (False, True):
            model = tiny_model(lora)
            if dtype is not None:
                model = model.to(dtype)
            w = wrapper(model, chunk=5)
            if checkpointing:
                w.train()
                w.gradient_checkpointing_enable()
            fwd = spark_oom.lean_forward if lean else self.original
            lp, out = fwd(w, seq, n, attention_mask=mask, temperature=temperature, return_output=True,
                          compute_entropy=compute_entropy, entropy_requires_grad=False)
            entry = {"lp": lp * action_mask}
            if compute_entropy:
                entry["ent"] = out["entropy"] * mask
            if grads:
                (lp * action_mask).sum().backward()
                entry["grads"] = {k: p.grad.clone() for k, p in model.named_parameters()
                                  if p.requires_grad and p.grad is not None}
            results.append(entry)
        up, lean = results
        torch.testing.assert_close(lean["lp"], up["lp"], atol=atol, rtol=rtol)
        if compute_entropy:
            torch.testing.assert_close(lean["ent"].float(), up["ent"].float(), atol=atol, rtol=rtol)
        if grads:
            self.assertEqual(set(lean["grads"]), set(up["grads"]))
            for k in up["grads"]:
                torch.testing.assert_close(lean["grads"][k].float(), up["grads"][k].float(),
                                           atol=atol, rtol=rtol, msg=k)
        return lean

    def test_full_model(self):
        self.compare(lora=False, compute_entropy=True)

    def test_lora_with_temperature(self):
        self.compare(lora=True, compute_entropy=True, temperature=0.7)

    def test_lora_with_gradient_checkpointing(self):
        """Upstream trains with gradient checkpointing on (its default)."""
        self.compare(lora=True, compute_entropy=True, checkpointing=True)

    def test_bf16_like_the_spark(self):
        """bf16 weights as on the GB10; the log-probs agree to bf16 rounding."""
        lean = self.compare(lora=True, compute_entropy=True, grads=False, dtype=torch.bfloat16,
                            atol=0.06, rtol=0.03)
        self.assertEqual(lean["lp"].dtype, torch.float32)

    def test_lora_reference_adapter_off(self):
        """The adapter-off reference pass (trainer_upstream) gives upstream's numbers too."""
        seq, mask, n = padded_batch()
        outs = []
        for fwd in (self.original, spark_oom.lean_forward):
            model = tiny_model(lora=True)
            w = wrapper(model, chunk=5)
            with torch.no_grad(), model.disable_adapter():
                outs.append(fwd(w, seq, n, attention_mask=mask) * mask[:, -n:])
        torch.testing.assert_close(outs[1], outs[0], atol=1e-5, rtol=1e-4)

    def test_padding_columns_are_zero_and_no_full_logits(self):
        seq, mask, n = padded_batch()
        w = wrapper(tiny_model(lora=False), chunk=5)
        lp, out = spark_oom.lean_forward(w, seq, n, attention_mask=mask, return_output=True,
                                         compute_entropy=True, entropy_requires_grad=False)
        self.assertIsNone(out["logits"])
        # action columns are log_probs[:, -n-1:-1]; row 0's padding is cols 17-18 of those
        self.assertTrue(torch.all(lp[0, -2:] == 0))
        self.assertTrue(torch.all(lp[0, -3] != 0))   # col 16 is row 0's last real token

    def test_unsupported_cases_use_upstream(self):
        seq, mask, n = padded_batch()
        w = wrapper(tiny_model(lora=False), chunk=5)
        sentinel = object()
        spark_oom._ORIGINAL_FORWARD = mock.Mock(return_value=sentinel)
        self.assertIs(spark_oom.lean_forward(w, seq, n, attention_mask=mask, compute_entropy=True,
                                             entropy_requires_grad=True), sentinel)
        w.sequence_parallel_size = 2
        self.assertIs(spark_oom.lean_forward(w, seq, n, attention_mask=mask), sentinel)

    def test_install_is_idempotent_and_can_be_turned_off(self):
        with mock.patch.dict(os.environ, {"E10_LEAN_FORWARD": "0"}):
            self.assertFalse(spark_oom.install_lean_forward())
            self.assertIs(HFModelWrapper.forward, self.original)
        self.assertTrue(spark_oom.install_lean_forward())
        self.assertTrue(spark_oom.install_lean_forward())
        self.assertIs(HFModelWrapper.forward, spark_oom.lean_forward)
        self.assertIs(spark_oom._ORIGINAL_FORWARD, self.original)


if __name__ == "__main__":
    unittest.main()
