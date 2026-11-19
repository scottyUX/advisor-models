"""Offline test of keep_models_resident() (E10 on the DGX Spark, unified memory).

Upstream's inference client and actor group are replaced by stand-ins with the
same method names; the test checks that sleep/wake and offload/backload become
no-ops, that is_sleeping() reports awake, that a missing upstream method fails
loudly, and that applying twice is harmless. No GPU, Ray, SkyRL or model.

Run: python -m unittest discover -s tests -p "test_e10_spark_resident.py"
"""

import asyncio
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODULE = "advisor_models.swe_smith.spark_resident"


def _upstream():
    calls = []

    class RemoteInferenceClient:
        async def sleep(self, level=2, tags=None):
            calls.append("sleep")

        async def wake_up(self, tags=None):
            calls.append("wake_up")

        async def sleep_for_weight_sync(self, offload_kv=True):
            calls.append("sleep_for_weight_sync")

        async def wake_for_weight_sync(self, tags):
            calls.append("wake_for_weight_sync")

        async def is_sleeping(self):
            return True

    class PPORayActorGroup:
        def offload_to_cpu(self, nonblocking=False, offload_optimizer=True, offload_model=True):
            calls.append("offload_to_cpu")

        def backload_to_gpu(self, nonblocking=False, backload_optimizer=True, backload_model=True):
            calls.append("backload_to_gpu")

    names = ["loguru", "skyrl", "skyrl.backends", "skyrl.backends.skyrl_train",
             "skyrl.backends.skyrl_train.inference_servers",
             "skyrl.backends.skyrl_train.inference_servers.remote_inference_client",
             "skyrl.backends.skyrl_train.workers", "skyrl.backends.skyrl_train.workers.worker"]
    mods = {n: types.ModuleType(n) for n in names}
    mods["loguru"].logger = types.SimpleNamespace(info=lambda *a, **k: None)
    mods["skyrl.backends.skyrl_train.inference_servers.remote_inference_client"].RemoteInferenceClient = \
        RemoteInferenceClient
    mods["skyrl.backends.skyrl_train.workers.worker"].PPORayActorGroup = PPORayActorGroup
    return mods, RemoteInferenceClient, PPORayActorGroup, calls


class TestKeepResident(unittest.TestCase):
    def setUp(self):
        self.mods, self.Client, self.Group, self.calls = _upstream()
        self.patch = mock.patch.dict(sys.modules, self.mods)
        self.patch.start()
        sys.modules.pop(MODULE, None)
        self.m = importlib.import_module(MODULE)

    def tearDown(self):
        sys.modules.pop(MODULE, None)
        self.patch.stop()

    def test_sleep_wake_and_offload_do_nothing(self):
        self.m.keep_models_resident()
        client, group = self.Client(), self.Group()

        async def drive():
            await client.sleep(level=1)
            await client.wake_up(tags=["weights"])
            await client.sleep_for_weight_sync(offload_kv=True)
            await client.wake_for_weight_sync(tags=["kv_cache"])
            return await client.is_sleeping()

        self.assertFalse(asyncio.run(drive()))
        self.assertIsNone(group.offload_to_cpu())
        self.assertIsNone(group.backload_to_gpu(backload_optimizer=False, backload_model=True))
        self.assertEqual(group.offload_to_cpu(nonblocking=True), [])
        self.assertEqual(self.calls, [])

    def test_twice_is_harmless(self):
        self.m.keep_models_resident()
        self.m.keep_models_resident()
        self.assertFalse(asyncio.run(self.Client().is_sleeping()))

    def test_missing_upstream_method_fails(self):
        del self.Client.sleep_for_weight_sync
        with self.assertRaises(RuntimeError):
            self.m.keep_models_resident()


if __name__ == "__main__":
    unittest.main()
