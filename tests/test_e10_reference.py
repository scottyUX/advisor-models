"""Offline test of E10's adapter-off reference model (spec §2).

Upstream SkyRL's base classes are replaced by minimal stand-ins that do what
upstream 3e8700f does at the points E10 hooks in: build_models makes actor
groups through ``skyrl.train.trainer.PPORayActorGroup`` (the worker class is
the fourth argument) and fwd_logprobs_values_reward fills
``base_action_log_probs`` only from a reference group. Checks that E10 builds
no reference group, uses the adapter-off policy worker, computes the
reference log-probs with the adapter disabled, and refuses settings it does
not support. No GPU, Ray, SkyRL or model.

Run: python -m unittest discover -s tests -p "test_e10_reference.py"
"""

import contextlib
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODULE = "advisor_models.swe_smith.trainer_upstream"


def ns(**kw):
    return types.SimpleNamespace(**kw)


class FakeGroup:
    def __init__(self, cfg, num_nodes, num_gpus_per_node, ray_actor_type, **kwargs):
        self.worker = ray_actor_type


class StubTrainer:
    """What upstream's RayPPOTrainer does where E10 hooks in."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.ref_model = None

    def build_models(self, PolicyWorker, CriticWorker, RefWorker):
        upstream = sys.modules["skyrl.train.trainer"]
        self.policy_model = upstream.PPORayActorGroup(self.cfg.trainer, 1, 1, PolicyWorker)
        if self.cfg.trainer.algorithm.use_kl_loss:
            self.ref_model = upstream.PPORayActorGroup(self.cfg.trainer, 1, 1, RefWorker)

    def fwd_logprobs_values_reward(self, training_input):
        training_input["base_action_log_probs"] = None if self.ref_model is None else "upstream-ref"
        training_input["action_log_probs"] = "policy"
        return training_input


class StubPolicyWorker:
    def forward(self, data, loss_fn=None, loss_fn_config=None, return_per_token_outputs=True):
        return {"adapter_disabled": self.model.model.disabled, "data": data}


class FakePeft:
    def __init__(self):
        self.disabled = False

    @contextlib.contextmanager
    def disable_adapter(self):
        self.disabled = True
        try:
            yield
        finally:
            self.disabled = False


class Batch(dict):
    metadata = {}

    def get(self, key, default=None):
        return super().get(key, default)

    def select(self, keys, metadata_keys):
        return {"keys": list(keys), "metadata_keys": list(metadata_keys)}


def _cfg(**over):
    algo = ns(use_kl_loss=True, use_kl_in_reward=False)
    trainer = ns(strategy="fsdp", update_ref_every_epoch=False, algorithm=algo,
                 policy=ns(model=ns(lora=ns(rank=32))))
    for k, v in over.items():
        if k in ("use_kl_loss", "use_kl_in_reward"):
            setattr(algo, k, v)
        elif k == "rank":
            trainer.policy.model.lora.rank = v
        else:
            setattr(trainer, k, v)
    return ns(trainer=trainer)


def _stubs():
    names = ["ray", "loguru", "skyrl", "skyrl.train", "skyrl.train.trainer", "skyrl.train.utils",
             "skyrl.train.utils.callbacks", "skyrl.backends", "skyrl.backends.skyrl_train",
             "skyrl.backends.skyrl_train.workers", "skyrl.backends.skyrl_train.workers.fsdp",
             "skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker"]
    mods = {n: types.ModuleType(n) for n in names}
    mods["ray"].remote = lambda **kw: (lambda cls: cls)
    mods["loguru"].logger = ns(info=lambda *a, **k: None)
    up = mods["skyrl.train.trainer"]
    up.RayPPOTrainer = StubTrainer
    up.PPORayActorGroup = FakeGroup
    up.SAMPLE_SUPPORT_FIELD = "sample_support"
    up.loss_fn_outputs_to_tensor = lambda outputs, key: [o[key] for o in outputs]
    mods["skyrl.train"].trainer = up
    mods["skyrl.train.utils.callbacks"].TrainingCallback = object
    mods["skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker"].FSDPPolicyWorkerBase = StubPolicyWorker
    return mods


class TestAdapterOffReference(unittest.TestCase):
    def setUp(self):
        self.patch = mock.patch.dict(sys.modules, _stubs())
        self.patch.start()
        sys.modules.pop(MODULE, None)
        self.m = importlib.import_module(MODULE)
        self.up = sys.modules["skyrl.train.trainer"]

    def tearDown(self):
        sys.modules.pop(MODULE, None)
        self.patch.stop()

    def test_no_reference_group_and_adapter_off_worker(self):
        t = self.m.E10RayPPOTrainer(_cfg())
        t.build_models("upstream-policy-worker", "critic-worker", "upstream-ref-worker")
        self.assertIsNone(t.ref_model)
        self.assertEqual(t.policy_model.worker.__name__, "E10FSDPPolicyWorker")
        self.assertIs(self.up.PPORayActorGroup, FakeGroup)  # restored after build_models

    def test_worker_forward_disables_adapter_only_for_reference(self):
        worker_cls = self.m.adapter_off_policy_worker()
        w = worker_cls()
        w.model = ns(model=FakePeft())
        self.assertTrue(w.forward("batch", reference=True)["adapter_disabled"])
        self.assertFalse(w.forward("batch")["adapter_disabled"])
        self.assertFalse(w.model.model.disabled)  # re-enabled after the reference forward
        w.model = ns(model=object())
        with self.assertRaises(RuntimeError):
            w.forward("batch", reference=True)

    def test_reference_log_probs_from_policy_with_adapter_off(self):
        t = self.m.E10RayPPOTrainer(_cfg())
        t.build_models("p", "c", "r")
        calls = []

        def forward(model, data, **kwargs):
            calls.append((model, data, kwargs))
            return ns(loss_fn_outputs=[{"logprobs": -0.5}, {"logprobs": -0.7}, {"logprobs": -9.9}])

        t.dispatch = ns(forward=forward, empty_cache=lambda *a: None)
        batch = Batch(sequences=[[1], [2]], attention_mask=[[1], [1]])
        out = t.fwd_logprobs_values_reward(batch)
        self.assertEqual(calls[0][0], "policy")
        self.assertEqual(calls[0][2], {"reference": True})
        self.assertEqual(calls[0][1]["keys"], ["sequences", "attention_mask"])
        self.assertEqual(out["base_action_log_probs"], [-0.5, -0.7])  # padding rows dropped
        self.assertEqual(out["action_log_probs"], "policy")

    def test_no_kl_loss_no_reference_forward(self):
        t = self.m.E10RayPPOTrainer(_cfg(use_kl_loss=False))
        t.build_models("p", "c", "r")
        t.dispatch = ns(forward=lambda *a, **k: self.fail("no reference forward expected"))
        self.assertIsNone(t.fwd_logprobs_values_reward(Batch(sequences=[[1]]))["base_action_log_probs"])

    def test_refuses_unsupported_settings(self):
        for over in [dict(strategy="megatron"), dict(rank=0), dict(use_kl_in_reward=True),
                     dict(update_ref_every_epoch=True)]:
            with self.subTest(over=over), self.assertRaises(ValueError):
                self.m.E10RayPPOTrainer(_cfg(**over)).build_models("p", "c", "r")


if __name__ == "__main__":
    unittest.main()
