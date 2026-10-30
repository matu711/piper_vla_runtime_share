"""pi0 Runtime wire-contract tests without weights, ROS or controllers."""
import unittest
from unittest import mock
import numpy as np
import yaml
from piper_vla_dual_runtime.profiles import DualProfile


class Pi0ProfileTests(unittest.TestCase):
    def runtime_contract(self):
        profile = DualProfile("dual_pi0_joint_v1", False, "standard", True)
        return dict(model_family="pi0", compatibility_profile=profile.name,
                    contract=profile.contract, training_actions=profile.training_actions)

    def test_pi0_runtime_uses_its_explicit_wire_contract_without_model_or_controller(self):
        import test_dual_runtime as runtime_tests
        harness = runtime_tests.RuntimeTests()
        harness.setUp()
        self.addCleanup(harness.doCleanups)
        fixture = {'manifest': self.runtime_contract()}
        (harness.path / 'manifest.yaml').write_text(yaml.safe_dump(fixture['manifest']))
        with mock.patch.dict(__import__('os').environ, RUNTIME_CONFIG=str(harness.path / 'runtime.yaml')):
            harness.runtime = runtime_tests.Runtime()
        self.addCleanup(harness.runtime.client.close)
        generation = harness.start()
        self.assertFalse(harness.runtime.profile.pi05)
        self.assertEqual(harness.runtime.policy_contract['normalization'], 'standard')
        harness.runtime.client.metadata_validator(harness.runtime.policy_contract)
        with self.assertRaises(ValueError):
            harness.runtime.client.metadata_validator(dict(harness.runtime.policy_contract, normalization='quantile'))
        actions = np.tile(np.arange(14) / 100, (50, 1))
        harness.runtime.chunks.stage(actions, generation)
        harness.runtime.tick()
        np.testing.assert_equal(harness.runtime.pubs['selected_action'].messages[-1].data, actions[0])
        self.assertEqual(harness.runtime.pubs['approved_action'].messages, [])
