"""Optional external-source contract check; ROS/SDK imports are not executed."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import threading
import time
import types
import unittest
from unittest import mock

import numpy as np
import test_dual_runtime as runtime_tests
from ros_mocks import JointState, String, Float64MultiArray, Stamp

ROSPY = runtime_tests.ROSPY
state = runtime_tests.state


class ExternalControllerTests(unittest.TestCase):
    def setUp(self):
        started = time.monotonic()
        patch = mock.patch.object(ROSPY.Time, "now", lambda: Stamp(100. + time.monotonic() - started))
        patch.start()
        self.addCleanup(patch.stop)
        runtime_tests.RuntimeTests.setUp(self)
    start = runtime_tests.RuntimeTests.start

    def make_controller(self):
        # Execute the installed Controller class without its ROS/SDK imports
        # or constructor. Only per-arm hardware/safety objects are test doubles.
        import uuid
        source = Path(os.environ["PIPER_DUAL_CONTROLLER_SOURCE"])
        tree = ast.parse(source.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "VlaDualPiperController")
        namespace = dict(threading=threading, uuid=uuid, rospy=ROSPY, math=__import__("math"),
                         time=time, json=json, JointState=JointState,
                         TriggerResponse=lambda **kwargs: types.SimpleNamespace(**kwargs))
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)
        controller = namespace[cls.name].__new__(namespace[cls.name])
        patch = mock.patch.object(ROSPY, "logwarn_throttle", create=True)
        patch.start()
        self.addCleanup(patch.stop)
        controller.lock = threading.RLock()
        controller.hold_requested, controller.fault_latched = True, False
        controller.session_token = None
        controller.states, controller.state_received = {}, {}
        controller.state_timeout_s = .5
        controller.max_delta = {side: [1.]*6 for side in ("left", "right")}  # Mock only.
        feedback_spec = importlib.util.spec_from_file_location("feedback_fixture", source.parent / "dual_feedback.py")
        feedback_module = importlib.util.module_from_spec(feedback_spec)
        feedback_spec.loader.exec_module(feedback_module)
        controller.feedback = feedback_module.FeedbackWatch()
        controller.state_pub = mock.Mock()
        controller.arms = {side: mock.Mock() for side in ("left", "right")}
        controller.safety_state_pub = mock.Mock()
        controller.safety_reason_pub = mock.Mock()
        controller.fault_pub = mock.Mock()
        controller.status_pub = mock.Mock(side_effect=None)
        controller.status_pub.publish.side_effect = lambda payload: self.runtime._controller_status(String(data=payload))
        for side, arm in controller.arms.items():
            arm.params.GetJointLimitParam.return_value = (-3., 3.)
            arm._apply_joint_saturation.side_effect = lambda action: list(action)
            controller.feedback.update(side, dict(schema=1, epoch="fixture", healthy=True,
                frames={key: dict(count=1, age_s=0., source_stamp=100.) for key in feedback_module.FRAMES}), .5)
            controller.state_callback(state(), side)
        return controller

    def connect_controller(self, controller):
        self.runtime.dry = False
        generation = self.start()
        actions = np.arange(700, dtype=float).reshape(50, 14) / 1000
        self.runtime.chunks.stage(actions, generation)
        self.runtime.controller_started = time.monotonic()
        with mock.patch.object(ROSPY, "ServiceProxy", return_value=lambda: controller.start(None)):
            self.runtime._controller_start(generation)
        controller.publish_status(None)
        return actions

    def test_actual_controller_callbacks_accept_first_and_all_actions(self):
        controller = self.make_controller()
        actions = self.connect_controller(controller)
        for _ in range(20):
            self.runtime.tick()
            controller.action_callback(self.runtime.pubs["approved_action"].messages[-1])
        for index, side in enumerate(("left", "right")):
            arm = controller.arms[side]
            self.assertEqual(arm.commit.call_count, 20)
            np.testing.assert_equal(arm.handle_joint_absolute.call_args_list[0].args[0],
                                    actions[0, index * 7:(index + 1) * 7])
        self.assertFalse(controller.hold_requested)
        controller.fault_pub.publish.assert_not_called()

    def test_stop_fault_and_restart_reject_old_tokens(self):
        for event in ("stop", "fault", "restart"):
            with self.subTest(event=event):
                self.setUp()
                controller = self.make_controller()
                self.connect_controller(controller)
                self.runtime.tick()
                old = self.runtime.pubs["approved_action"].messages[-1]
                for arm in controller.arms.values():
                    arm.commit.reset_mock()
                if event == "stop":
                    controller.stop(None)
                elif event == "fault":
                    controller.controller_fault("test_fault")
                else:
                    controller = self.make_controller()  # __init__ token/HOLD state.
                controller.action_callback(old)
                for arm in controller.arms.values():
                    arm.commit.assert_not_called()
                # Recovery requires explicit start and generates a new token.
                self.assertTrue(controller.start(None).success)
                self.assertNotEqual(controller.session_token, self.runtime.controller_token)
                controller.action_callback(old)
                for arm in controller.arms.values():
                    arm.commit.assert_not_called()

    def test_invalid_and_missing_tokens_are_silently_rejected(self):
        controller = self.make_controller()
        self.connect_controller(controller)
        self.runtime.tick()
        message = self.runtime.pubs["approved_action"].messages[-1]
        message.layout.dim[0].label = "controller_session:" + "0" * 32
        controller.action_callback(message)
        controller.action_callback(Float64MultiArray(data=[0.] * 14))
        for arm in controller.arms.values():
            arm.commit.assert_not_called()
        controller.fault_pub.publish.assert_not_called()

    def test_controller_restart_is_detected_with_fresh_observations(self):
        controller = self.make_controller()
        self.connect_controller(controller)
        self.runtime.tick()
        restarted = self.make_controller()
        restarted.publish_status(None)
        self.assertTrue(restarted.hold_requested)
        self.assertFalse(self.runtime.chunks.enabled)
        self.assertIsNone(self.runtime.controller_token)
        self.assertEqual(self.runtime.pubs["safety_state"].messages[-1].data, "HOLD")
        restarted.fault_pub.publish.assert_not_called()
        for arm in restarted.arms.values():
            arm.commit.assert_not_called()

    def test_periodic_status_holds_runtime_when_fault_notification_is_lost(self):
        for delivered in (True, False):
            with self.subTest(delivered=delivered):
                self.setUp()
                controller = self.make_controller()
                self.connect_controller(controller)
                self.runtime.tick()
                if delivered:
                    controller.fault_pub.publish.side_effect = lambda reason: self.runtime._controller_fault(String(data=reason))
                controller.stop(None)
                controller.publish_status(None)
                self.runtime.tick()
                self.assertFalse(self.runtime.chunks.enabled)
                self.assertIsNone(self.runtime.controller_token)
                self.assertIsNone(self.runtime.chunks.pending)
                self.assertFalse(self.runtime.chunks.active)
                self.assertEqual(self.runtime.pubs["safety_state"].messages[-1].data, "HOLD")
                for arm in controller.arms.values():
                    arm.commit.assert_not_called()
