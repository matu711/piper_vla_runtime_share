#!/usr/bin/env python3
"""Unit tests for PBL8 Runtime gripper saturation and publish flow."""
import math
import types
import unittest
from unittest import mock

from piper_vla_runtime import pbl8_gripper as pbl8

try:
    from piper_vla_runtime import node as runtime_node
except ImportError:
    # The host-side pure helper tests do not require ROS/OpenCV. The Docker
    # client image has those dependencies and runs the publish-flow tests too.
    runtime_node = None


class Pbl8GripperRuntimeTest(unittest.TestCase):
    def test_previous_hold_value_is_saturated_without_mutating_raw(self):
        raw = [0.001, -0.001, 0.001, 0.004, -0.005, 0.006, -1.0009]
        applied = pbl8.apply_pbl8_gripper_saturation(raw)
        self.assertEqual(raw[6], -1.0009)
        self.assertEqual(applied[:6], raw[:6])
        self.assertEqual(applied[6], -1.0)

    def test_positive_small_extrapolation_is_saturated(self):
        raw = [0.0] * 6 + [1.0009]
        applied = pbl8.apply_pbl8_gripper_saturation(raw)
        self.assertEqual(raw[6], 1.0009)
        self.assertEqual(applied[:6], raw[:6])
        self.assertEqual(applied[6], 1.0)

    def test_margin_boundaries_are_inclusive(self):
        self.assertEqual(
            pbl8.apply_pbl8_gripper_saturation([0.0] * 6 + [-1.01])[6],
            -1.0,
        )
        self.assertEqual(
            pbl8.apply_pbl8_gripper_saturation([0.0] * 6 + [1.01])[6],
            1.0,
        )

    def test_values_inside_runtime_range_are_unchanged(self):
        for gripper in (-1.0, -0.4, 0.0, 0.7, 1.0):
            raw = [0.1, 0.2, 0.3, 0.01, 0.02, 0.03, gripper]
            self.assertEqual(pbl8.apply_pbl8_gripper_saturation(raw), raw)

    def test_values_outside_margin_are_rejected(self):
        for gripper in (-1.010001, 1.010001):
            with self.assertRaisesRegex(ValueError, "outside saturation margin"):
                pbl8.apply_pbl8_gripper_saturation([0.0] * 6 + [gripper])

    def test_non_finite_values_are_rejected(self):
        for gripper in (math.nan, math.inf, -math.inf):
            with self.assertRaisesRegex(ValueError, "outside saturation margin"):
                pbl8.apply_pbl8_gripper_saturation([0.0] * 6 + [gripper])


@unittest.skipIf(runtime_node is None, "Runtime ROS dependencies are unavailable")
class Pbl8RuntimePublishFlowTest(unittest.TestCase):
    def make_runtime(self):
        runtime = runtime_node.Runtime.__new__(runtime_node.Runtime)
        runtime.chunks = runtime_node.ChunkQueue(1, 0, runtime_node.ActionContract(7), reset_ids=False, underrun_error=False)
        runtime.is_pbl8 = True
        runtime.pbl8_max_translation_delta_m = 0.003
        runtime.pbl8_max_rotation_delta_rad = 0.01
        runtime.enabled = True
        runtime._awaiting_initial_chunk = False
        runtime._inference_request = types.SimpleNamespace(clear=lambda: None)
        runtime.clear_chunks_count = 0
        runtime._clear_chunks = lambda: setattr(
            runtime, "clear_chunks_count", runtime.clear_chunks_count + 1
        )
        runtime.cfg = {
            "robot": {"approved_action_topic": "/vla/approved_action"},
            "safety": {"z_enabled": True},
            "runtime": {"dry_run": False},
        }
        for name in (
            "selected_pub", "z_pub", "target_pub", "safety_pub",
            "reason_pub", "hold_pub", "command_pub",
        ):
            setattr(runtime, name, object())
        runtime.local_target_z = lambda _action: None
        runtime.events = []
        runtime.hold_reasons = []
        runtime._publish_recorded = lambda _pub, topic, message: (
            runtime.events.append((topic, message.data))
        )
        def hold_chunk(reason):
            runtime.hold_reasons.append(reason)
            runtime.events.extend((
                ("/safety/state", "HOLD"),
                ("/safety/reason", reason),
                ("/vla/hold", True),
            ))
        runtime._hold_chunk = hold_chunk
        return runtime

    def test_selected_stays_raw_approved_is_saturated_and_execution_continues(self):
        runtime = self.make_runtime()
        raw = [0.001, -0.001, 0.001, 0.004, -0.005, 0.006, -1.0009]
        with mock.patch.object(runtime_node.rospy, "logwarn_throttle"), mock.patch.object(
            runtime_node.rospy, "logdebug"
        ):
            result = runtime._publish_chunk_action(1, 0, raw)
        events = dict(runtime.events)
        self.assertTrue(result)
        self.assertEqual(events["/vla/selected_action"], raw)
        self.assertEqual(events["/vla/approved_action"][:6], raw[:6])
        self.assertEqual(events["/vla/approved_action"][6], -1.0)
        self.assertEqual(events["/vla/hold"], False)
        self.assertEqual(runtime.hold_reasons, [])

    def test_outside_margin_keeps_pbl8_gripper_range_hold(self):
        runtime = self.make_runtime()
        raw = [0.0] * 6 + [-1.010001]
        with mock.patch.object(runtime_node.rospy, "logwarn"):
            result = runtime._publish_chunk_action(1, 0, raw)
        self.assertFalse(result)
        self.assertEqual(runtime.hold_reasons, ["pbl8_gripper_range"])
        events = dict(runtime.events)
        self.assertNotIn("/vla/approved_action", events)
        self.assertEqual(events["/safety/state"], "HOLD")
        self.assertEqual(events["/safety/reason"], "pbl8_gripper_range")
        self.assertTrue(events["/vla/hold"])
        self.assertFalse(runtime.enabled)
        self.assertEqual(runtime.clear_chunks_count, 1)

    def test_translation_limit_rejects_only_one_step_then_resumes(self):
        runtime = self.make_runtime()
        runtime.pbl8_max_translation_delta_m = 0.005
        actions = (
            [0.004, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            [0.006, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            [0.004, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        )
        results = []
        with mock.patch.object(runtime_node.rospy, "logwarn"), mock.patch.object(
            runtime_node.rospy, "logwarn_throttle"
        ), mock.patch.object(runtime_node.rospy, "logdebug"):
            for index, action in enumerate(actions):
                results.append(runtime._publish_chunk_action(1, index, action))

        selected = [
            list(data) for topic, data in runtime.events
            if topic == "/vla/selected_action"
        ]
        approved = [
            list(data) for topic, data in runtime.events
            if topic == "/vla/approved_action"
        ]
        states = [
            data for topic, data in runtime.events if topic == "/safety/state"
        ]
        reasons = [
            data for topic, data in runtime.events if topic == "/safety/reason"
        ]
        holds = [data for topic, data in runtime.events if topic == "/vla/hold"]

        self.assertEqual(results, [True, False, True])
        self.assertEqual(selected, list(actions))
        self.assertEqual(approved, [actions[0], actions[2]])
        self.assertEqual(states, ["APPROVED", "REJECTED", "APPROVED"])
        self.assertEqual(
            reasons, ["", "pbl8_translation_delta_limit", ""]
        )
        self.assertNotIn(True, holds)
        self.assertTrue(runtime.enabled)
        self.assertEqual(runtime.hold_reasons, [])
        self.assertEqual(runtime.clear_chunks_count, 0)

    def test_rotation_limit_is_step_reject_without_clip_or_hold(self):
        runtime = self.make_runtime()
        runtime.pbl8_max_rotation_delta_rad = 0.005
        rejected = [0.0, 0.0, 0.0, 0.006, 0.0, 0.0, -1.0]
        with mock.patch.object(runtime_node.rospy, "logwarn"), mock.patch.object(
            runtime_node.rospy, "logwarn_throttle"
        ):
            result = runtime._publish_chunk_action(1, 0, rejected)
        events = dict(runtime.events)
        self.assertFalse(result)
        self.assertEqual(events["/vla/selected_action"], rejected)
        self.assertNotIn("/vla/approved_action", events)
        self.assertEqual(events["/safety/state"], "REJECTED")
        self.assertEqual(events["/safety/reason"], "pbl8_rotation_delta_limit")
        self.assertNotIn("/vla/hold", events)
        self.assertTrue(runtime.enabled)
        self.assertEqual(runtime.hold_reasons, [])
        self.assertEqual(runtime.clear_chunks_count, 0)

if __name__ == "__main__":
    unittest.main()
