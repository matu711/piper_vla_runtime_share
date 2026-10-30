"""Controller status gates and watchdogs with deterministic ROS/steady clocks."""
import json
import types
import unittest
from unittest import mock

import numpy as np
from ros_mocks import Stamp, String
import test_dual_runtime as runtime_tests
from test_dual_runtime import ROSPY


class ControllerStatusTests(unittest.TestCase):
    def start(self):
        generation = runtime_tests.RuntimeTests.start(self)
        for message, *_ in self.runtime.observations.samples.values():
            message.header.stamp = Stamp(self.ros_now)
        return generation

    def setUp(self):
        self.now, self.ros_now = 1000., 100.
        for patch in (
            mock.patch("piper_vla_dual_runtime.node.time.monotonic", lambda: self.now),
            mock.patch.object(ROSPY.Time, "now", lambda: Stamp(self.ros_now)),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        runtime_tests.RuntimeTests.setUp(self)

    def advance(self, seconds):
        self.now += seconds
        self.ros_now += seconds

    def status(self, state="RUNNING", token=None, stamp=None):
        token = self.runtime.controller_token if token is None else token
        self.runtime._controller_status(String(data=json.dumps(dict(
            schema=1, state=state, token=token,
            stamp_ns=ROSPY.Time.now().to_nsec() if stamp is None else stamp))))

    def prepare(self, service=None):
        self.runtime.dry = False
        generation = self.start()
        self.runtime.chunks.stage(np.zeros((50, 14)), generation)
        self.runtime.controller_started = self.now
        response = lambda: types.SimpleNamespace(success=True, message="%032x" % generation)
        with mock.patch.object(ROSPY, "ServiceProxy", return_value=service or response):
            self.runtime._controller_start(generation)
        return generation

    def running(self):
        self.prepare()
        self.status()
        self.runtime.tick()
        self.assertEqual(self.runtime.execution_steps, 1)

    def assert_hold(self, reason):
        self.assertFalse(self.runtime.chunks.enabled)
        self.assertIsNone(self.runtime.controller_token)
        self.assertFalse(self.runtime.chunks.active)
        self.assertIsNone(self.runtime.chunks.pending)
        self.assertIsNone(self.runtime.controller_status)
        self.assertFalse(self.runtime.request.is_set())
        self.assertIn(reason, self.runtime.fault_reason)
        self.assertTrue(self.runtime.pubs["hold"].messages[-1].data)
        self.assertEqual(self.runtime.pubs["safety_state"].messages[-1].data, "HOLD")

    def test_start_requires_ack_and_matching_running_status_before_selection(self):
        self.prepare()
        self.runtime.tick()
        self.assertEqual(self.runtime.execution_steps, 0)
        self.assertIsNotNone(self.runtime.chunks.pending)
        self.assertEqual(self.runtime.chunks.index, 0)
        self.status(state="HOLD", token="")  # pre-start snapshot arriving after ACK
        self.runtime.tick()
        self.assertTrue(self.runtime.chunks.enabled)
        self.assertEqual(self.runtime.execution_steps, 0)
        self.advance(.05)
        self.status()
        self.runtime.tick()
        self.assertEqual(self.runtime.execution_steps, 1)
        self.assertEqual(self.runtime.chunks.index, 1)

    def test_status_before_ack_is_retained_and_old_hold_after_it_is_ignored(self):
        old_stamp = ROSPY.Time.now().to_nsec()
        def start_service():
            self.advance(.01)
            self.status(token="1".zfill(32))
            return types.SimpleNamespace(success=True, message="1".zfill(32))
        self.prepare(start_service)
        self.status("HOLD", token="", stamp=old_stamp)
        self.runtime.tick()
        self.assertEqual(self.runtime.execution_steps, 1)

    def test_hold_before_runtime_or_controller_start_is_not_a_fault(self):
        self.status("HOLD", token="")
        self.assertIsNone(self.runtime.fault_reason)
        self.start()
        self.status("HOLD", token="")
        self.runtime.tick()  # No pending chunk: no Controller start yet.
        self.assertTrue(self.runtime.chunks.enabled)
        self.assertIsNone(self.runtime.controller_started)

    def test_ack_without_status_has_finite_deadline(self):
        self.prepare()
        self.advance(2.01)
        self.runtime.tick()
        self.assert_hold("controller_status_start_timeout")
        self.assertEqual(self.runtime.execution_steps, 0)

    def test_wrong_token_during_start_does_not_authorize_actions(self):
        self.prepare()
        self.status(token="b" * 32)
        self.runtime.tick()
        self.assertEqual(self.runtime.execution_steps, 0)
        self.advance(2.01)
        self.runtime.tick()
        self.assert_hold("controller_status_start_timeout")

    def test_fault_hold_token_loss_and_mismatch_stop_active_chunks(self):
        for state, token, reason in (("FAULT", None, "fault"), ("HOLD", None, "hold"),
                                     ("HOLD", "", "token_mismatch"),
                                     ("RUNNING", "b" * 32, "token_mismatch")):
            with self.subTest(state=state, token=token):
                self.running()
                sent = self.runtime.execution_steps
                self.advance(.05)
                self.status(state, token)
                self.assert_hold("controller_status_" + reason)
                self.runtime.tick()
                self.assertEqual(self.runtime.execution_steps, sent)

    def test_silence_and_source_age_are_independent_watchdogs(self):
        for clock in ("steady", "ros", "both"):
            with self.subTest(clock=clock):
                self.running()
                if clock in ("steady", "both"):
                    self.now += .51
                if clock in ("ros", "both"):
                    self.ros_now += .51
                self.runtime.tick()
                self.assert_hold("controller_status_timeout")

    def test_replays_out_of_order_stale_and_future_packets_cannot_refresh(self):
        self.running()
        original = self.runtime.controller_status
        self.advance(.2)
        for stamp in (original[0], original[0] - 1, original[0] - 10**9,
                      ROSPY.Time.now().to_nsec() + 10**9):
            self.status("HOLD", token="", stamp=stamp)
            self.assertEqual(self.runtime.controller_status, original)
        self.advance(.31)
        self.runtime.tick()
        self.assert_hold("controller_status_timeout")

    def test_stale_matching_status_cannot_open_start_gate(self):
        self.prepare()
        self.advance(.6)
        self.status(stamp=ROSPY.Time.now().to_nsec() - 600000000)
        self.runtime.tick()
        self.assertEqual(self.runtime.execution_steps, 0)
        self.assertFalse(self.runtime.controller_status_confirmed)

    def test_stop_restart_drops_previous_status_and_requires_new_token(self):
        self.running()
        old_token = self.runtime.controller_token
        self.runtime.stop(None)
        self.assertIsNone(self.runtime.controller_status)
        self.prepare()
        self.assertNotEqual(self.runtime.controller_token, old_token)
        self.status(token=old_token)
        self.runtime.tick()
        self.assertEqual(self.runtime.execution_steps, 0)
        self.advance(.05)
        self.status()
        self.runtime.tick()
        self.assertEqual(self.runtime.execution_steps, 1)

    def test_invalid_status_and_frozen_ros_clock_fail_closed(self):
        self.running()
        self.runtime._controller_status(String(data="not json"))
        self.assert_hold("controller_status:")
        self.running()
        original = self.runtime.controller_status
        self.now += .3
        self.status()  # Same ROS stamp: cannot refresh the steady-clock age.
        self.assertEqual(self.runtime.controller_status, original)
        self.now += .21
        self.runtime.tick()
        self.assert_hold("controller_status_timeout")

    def test_vla_timeout_continues_but_controller_silence_stops(self):
        self.running()
        token = self.runtime.controller_token
        self.runtime._communication_error("policy_timeout")
        self.advance(.05)
        self.status()
        self.runtime.tick()
        self.assertEqual(self.runtime.controller_token, token)
        self.assertEqual(self.runtime.execution_steps, 2)
        self.advance(.51)
        self.runtime.tick()
        self.assert_hold("controller_status_timeout")

    def test_late_fresh_status_cannot_hide_outage_before_next_tick(self):
        self.running()
        self.advance(.51)
        self.status()
        self.assert_hold("controller_status_timeout")

    def test_matching_fault_before_first_action_still_holds(self):
        self.prepare()
        self.status("FAULT")
        self.assert_hold("controller_status_fault")
        self.assertEqual(self.runtime.execution_steps, 0)

    def test_late_first_running_status_cannot_hide_start_deadline(self):
        self.prepare()
        self.advance(2.01)
        self.status()
        self.assert_hold("controller_status_start_timeout")
        self.assertEqual(self.runtime.execution_steps, 0)

    def test_backward_ros_clock_is_not_healthy(self):
        self.running()
        self.ros_now -= 1.
        self.runtime.tick()
        self.assert_hold("controller_status_timeout")
