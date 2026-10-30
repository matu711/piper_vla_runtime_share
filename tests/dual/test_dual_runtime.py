"""CPU-only tests; ROS API and model inference are mocked, not started."""
import ast
import dataclasses
import importlib.util
import json
import os
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import yaml
from mcap.reader import make_reader
from mcap_ros1.decoder import DecoderFactory
from ros_mocks import install, JointState, CompressedImage, Float64MultiArray, Bool, String, Stamp

ROSPY = install()
from piper_vla_dual_runtime.core import Chunks, Observations, JOINT_NAMES, DELTA_MASK, POLICY_CONTRACT, validate_contract
from piper_vla_common.checkpoint_package import runtime_manifest
from piper_vla_dual_runtime.node import Runtime
from piper_vla_dual_runtime.policy_transforms import IMAGE_KEYS, STATE_KEY
from test_common import DualInputs, DualOutputs
from piper_vla_dual_runtime.runtime_recorder import RuntimeRecorder

ROOT = Path(__file__).resolve().parents[2]


def state(values=None, stamp=100.0):
    message = JointState()
    message.name, message.position = list(JOINT_NAMES), list(values if values is not None else [0.0] * 6 + [0.04])
    message.header.stamp = Stamp(stamp)
    return message


class CoreTests(unittest.TestCase):
    def test_state_order_and_three_cameras(self):
        obs = Observations(0.5, 0.1)
        left = state([1, 2, 3, 4, 5, 6, 0.03])
        left.name.reverse()
        left.position.reverse()
        obs.update("left", left, 10)
        obs.update("right", state([7, 8, 9, 10, 11, 12, 0.04]), 10)
        for camera in ("front", "left_wrist", "right_wrist"):
            obs.update(camera, CompressedImage(), 10)
        values, images = obs.snapshot(10.1, 100.1)
        np.testing.assert_array_almost_equal(values, [1, 2, 3, 4, 5, 6, .03, 7, 8, 9, 10, 11, 12, .04])
        self.assertEqual(set(images), {"front", "left_wrist", "right_wrist"})
        self.assertEqual(values.dtype, np.float32)
        with self.assertRaisesRegex(ValueError, "stale"):
            obs.snapshot(11, 100.1)
        with self.assertRaisesRegex(ValueError, "post_start"):
            obs.snapshot(10.1, 100.1, dict(obs.versions))

    def test_observation_missing_skew_names_nan(self):
        obs = Observations(.5, .1)
        with self.assertRaisesRegex(ValueError, "missing"):
            obs.snapshot(10, 100)
        for name in ("left", "right", "front", "left_wrist", "right_wrist"):
            obs.update(name, state() if name in ("left", "right") else CompressedImage(), 10)
        obs.samples["front"][0].header.stamp = Stamp(99.8)
        with self.assertRaisesRegex(ValueError, "skew"):
            obs.snapshot(10, 100)
        obs.samples["front"][0].header.stamp = Stamp(100)
        obs.samples["left"][0].name = ["bad"] * 7
        with self.assertRaisesRegex(ValueError, "names"):
            obs.snapshot(10, 100)
        obs.samples["left"][0].name = list(JOINT_NAMES)
        obs.samples["left"][0].position[0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            obs.snapshot(10, 100)

    def test_chunk_prefetch_order_and_no_replacement(self):
        chunks = Chunks(3, 1)
        generation = chunks.start()
        self.assertIsNone(chunks.select())
        data = np.arange(700).reshape(50, 14)
        self.assertTrue(chunks.stage(data, generation))
        self.assertFalse(chunks.select()[3])
        self.assertTrue(chunks.select()[3])
        chunks.stage(data + 1000, generation)
        with self.assertRaisesRegex(ValueError, "pending"):
            chunks.stage(data, generation)
        self.assertEqual(chunks.select()[:2], (1, 2))
        selected = chunks.select()
        self.assertEqual(selected[:2], (2, 0))
        np.testing.assert_equal(selected[2], data[0] + 1000)

    def test_chunk_underrun_discard_and_bad_shapes(self):
        chunks = Chunks(1, 0)
        generation = chunks.start()
        for shape in ((50, 7), (20, 14), (50, 32)):
            with self.assertRaises(ValueError):
                chunks.stage(np.zeros(shape), generation)
        data = np.zeros((50, 14))
        data[-1, -1] = np.inf
        with self.assertRaises(ValueError):
            chunks.stage(data, generation)
        chunks.stage(np.zeros((50, 14)), generation)
        chunks.select()
        with self.assertRaisesRegex(ValueError, "underrun"):
            chunks.select()
        chunks.stop()
        self.assertFalse(chunks.stage(np.zeros((50, 14)), generation))
        chunks.start()
        self.assertFalse(chunks.stage(np.zeros((50, 14)), generation))

    def test_contract_requires_absolute_physical_output(self):
        validate_contract(POLICY_CONTRACT)
        for key, value in (("action_type", "joint_delta"), ("action_dim", 7), ("action_horizon", 20)):
            with self.assertRaises(ValueError):
                validate_contract(dict(POLICY_CONTRACT, **{key: value}))


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        manifest = runtime_manifest(dict(robot="bimanual", family="pi05", config_name="test_policy",
            repo_id="local/test", model=dict(action_horizon=50, action_dim=32),
            contract=dict(POLICY_CONTRACT), delta_mask=list(DELTA_MASK)))
        (self.path / "manifest.yaml").write_text(yaml.safe_dump(manifest))
        config = yaml.safe_load((ROOT / "configs/runtime.dual.yaml").read_text())
        config["model"]["manifest"] = str(self.path / "manifest.yaml")
        config["recording"]["enabled"] = False
        (self.path / "runtime.yaml").write_text(yaml.safe_dump(config))
        with mock.patch.dict(os.environ, RUNTIME_CONFIG=str(self.path / "runtime.yaml")):
            self.runtime = Runtime()
        self.addCleanup(lambda: self.runtime.client.close())

    def start(self):
        response = self.runtime.start(types.SimpleNamespace(prompt="pick with both arms"))
        self.assertTrue(response.success, response.message)
        for name in ("left", "right"):
            self.runtime._callback(name)(state())
        for i, name in enumerate(("front", "left_wrist", "right_wrist")):
            image = np.full((30, 50, 3), [10 + i, 20 + i, 30 + i], np.uint8)
            ok, encoded = cv2.imencode(".png", image)
            self.assertTrue(ok)
            message = CompressedImage()
            message.data = encoded.tobytes()
            self.runtime._callback(name)(message)
        return self.runtime.chunks.generation

    def controller_status(self, state="RUNNING", token=None, stamp_ns=None):
        self.runtime._controller_status(String(data=json.dumps(dict(
            schema=1, state=state,
            token=self.runtime.controller_token if token is None else token,
            stamp_ns=ROSPY.Time.now().to_nsec() if stamp_ns is None else stamp_ns))))

    def test_images_request_and_dry_run_no_hardware_or_release(self):
        generation = self.start()
        state_vector, images = self.runtime._snapshot()
        obs = self.runtime._build_observation(state_vector, images, "test")
        converted = DualInputs()(obs)
        self.assertEqual(converted["state"].shape, (14,))
        self.assertEqual(len(converted["image"]), 3)
        for i, name in enumerate(("front", "left_wrist", "right_wrist")):
            np.testing.assert_equal(obs[IMAGE_KEYS[name]][112, 112], [30 + i, 20 + i, 10 + i])
        actions = np.tile(np.arange(14) / 100, (50, 1))
        self.runtime.chunks.stage(actions, generation)
        self.runtime.tick()
        self.assertEqual(self.runtime.pubs["approved_action"].messages, [])
        self.assertTrue(all(item.data for item in self.runtime.pubs["hold"].messages))
        np.testing.assert_equal(self.runtime.pubs["selected_action"].messages[-1].data, actions[0])

    def test_stop_invalidates_inflight_response(self):
        generation = self.start()
        self.runtime.chunks.stage(np.zeros((50, 14)), generation)
        self.runtime.stop(None)
        self.assertFalse(self.runtime.chunks.enabled)
        self.assertFalse(self.runtime.chunks.stage(np.zeros((50, 14)), generation))
        self.assertIsNone(self.runtime.chunks.pending)
        self.assertTrue(self.runtime.pubs["hold"].messages[-1].data)

    def test_active_observation_loss_stops_without_command(self):
        generation = self.start()
        self.runtime.chunks.stage(np.zeros((50, 14)), generation)
        del self.runtime.observations.samples["right_wrist"]
        self.runtime.chunks.awaiting_initial = False
        self.runtime.tick()
        self.assertFalse(self.runtime.chunks.enabled)
        self.assertEqual(self.runtime.pubs["selected_action"].messages, [])

    def test_underrun_and_controller_fault_latch(self):
        generation = self.start()
        self.runtime.chunks.length = 1
        self.runtime.chunks.stage(np.zeros((50, 14)), generation)
        self.runtime.tick()
        self.runtime.tick()
        self.assertFalse(self.runtime.chunks.enabled)
        self.assertIn("underrun", self.runtime.pubs["safety_reason"].messages[-1].data)

    def test_live_packet_is_absolute_without_state_addition(self):
        self.runtime.dry = False
        generation = self.start()
        actions = np.tile([.2] * 6 + [.04] + [.3] * 6 + [.05], (50, 1))
        self.runtime.chunks.stage(actions, generation)
        self.runtime.controller_started = time.monotonic()
        self.runtime._controller_start(generation)
        self.controller_status()
        self.runtime.tick()
        np.testing.assert_equal(self.runtime.pubs["approved_action"].messages[-1].data, actions[0])
        self.runtime._controller_fault(String(data="right_fault"))
        self.assertFalse(self.runtime.chunks.enabled)
        self.assertIsNone(self.runtime.chunks.pending)

    def test_controller_ack_delay_preserves_first_action_and_prefetch(self):
        self.runtime.dry = False
        generation = self.start()
        actions = np.arange(700, dtype=float).reshape(50, 14) / 1000
        self.runtime.chunks.stage(actions, generation)
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def delayed():
            entered.set()
            release.wait(2)
            return types.SimpleNamespace(success=True, message="b" * 32)

        with mock.patch.object(ROSPY, "ServiceProxy", return_value=delayed):
            self.runtime.tick()
            self.assertTrue(entered.wait(1))
            for _ in range(3):
                self.runtime.tick()
            self.assertEqual(self.runtime.pubs["selected_action"].messages, [])
            self.assertIsNotNone(self.runtime.chunks.pending)
            release.set()
            deadline = time.monotonic() + 1
            while self.runtime.controller_token is None and time.monotonic() < deadline:
                time.sleep(.001)
        self.assertEqual(self.runtime.controller_token, "b" * 32)
        self.controller_status()
        self.runtime.request.clear()
        for _ in range(16):
            self.runtime.tick()
        sent = self.runtime.pubs["approved_action"].messages
        self.assertEqual(len(sent), 16)
        for index, message in enumerate(sent):
            np.testing.assert_equal(message.data, actions[index])
            self.assertEqual(message.layout.dim[0].label, "controller_session:" + "b" * 32)
        self.assertTrue(self.runtime.request.is_set())
        self.assertTrue(all(item.data for item in self.runtime.pubs["hold"].messages))

    def test_controller_ack_timeout_and_late_response_cannot_restart(self):
        self.runtime.dry = False
        generation = self.start()
        self.runtime.chunks.stage(np.zeros((50, 14)), generation)
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def delayed():
            entered.set()
            release.wait(2)
            return types.SimpleNamespace(success=True, message="f" * 32)

        with mock.patch.object(ROSPY, "ServiceProxy", return_value=delayed):
            self.runtime.tick()
            self.assertTrue(entered.wait(1))
            self.runtime.controller_started = time.monotonic() - 3
            self.runtime.tick()
            self.assertEqual(self.runtime.fault_reason, "controller_start_timeout")
            release.set()
            # Take the call lock to ensure the old service call has returned.
            with self.runtime.controller_call_lock:
                pass
        self.assertEqual(self.runtime.pubs["approved_action"].messages, [])
        self.assertIsNone(self.runtime.controller_token)
        self.assertFalse(self.runtime.chunks.enabled)

    def test_stop_restart_ignores_old_inflight_ack(self):
        self.runtime.dry = False
        generation = self.start()
        self.runtime.controller_started = time.monotonic()
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def delayed():
            entered.set()
            release.wait(2)
            return types.SimpleNamespace(success=True, message="c" * 32)

        with mock.patch.object(ROSPY, "ServiceProxy", return_value=delayed):
            worker = threading.Thread(target=self.runtime._controller_start, args=(generation,))
            worker.start()
            self.assertTrue(entered.wait(1))
            self.runtime.stop(None)
            new_generation = self.start()
            self.runtime.chunks.stage(np.zeros((50, 14)), new_generation)
            release.set()
            worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertIsNone(self.runtime.controller_token)
        self.assertEqual(self.runtime.pubs["approved_action"].messages, [])
        self.runtime.controller_started = time.monotonic()
        with mock.patch.object(ROSPY, "ServiceProxy", return_value=lambda: types.SimpleNamespace(
                success=True, message="d" * 32)):
            self.runtime._controller_start(new_generation)
        self.controller_status()
        self.runtime.tick()
        self.assertEqual(len(self.runtime.pubs["approved_action"].messages), 1)
        self.assertEqual(self.runtime.pubs["approved_action"].messages[0].layout.dim[0].label,
                         "controller_session:" + "d" * 32)

    def test_controller_rejects_bad_or_reused_ack(self):
        self.runtime.dry = False
        for success, token in ((False, "a" * 32), (True, "bad"), (True, "e" * 32)):
            with self.subTest(token=token):
                generation = self.start()
                self.runtime.controller_tokens_seen.add("e" * 32)
                self.runtime.controller_started = time.monotonic()
                with mock.patch.object(ROSPY, "ServiceProxy", return_value=lambda: types.SimpleNamespace(
                        success=success, message=token)):
                    self.runtime._controller_start(generation)
                self.assertFalse(self.runtime.chunks.enabled)
                self.assertEqual(self.runtime.fault_reason, "controller_start_invalid_ack")
                self.assertEqual(self.runtime.pubs["approved_action"].messages, [])

    def arm_execution_limit(self, limit):
        self.runtime.max_execution_steps = limit
        self.runtime.dry = False
        generation = self.start()
        self.runtime.controller_started = time.monotonic()
        with mock.patch.object(ROSPY, "ServiceProxy", return_value=lambda: types.SimpleNamespace(
                success=True, message="%032x" % generation)):
            self.runtime._controller_start(generation)
        self.assertIsNotNone(self.runtime.controller_token)
        self.controller_status()
        self.runtime.chunks.stage(np.zeros((50, 14)), generation)
        self.runtime.request.clear()
        return generation

    def run_execution_ticks(self, ticks):
        prefetch = 0
        for _ in range(ticks):
            self.runtime.tick()
            if self.runtime.request.is_set() and self.runtime.chunks.enabled:
                prefetch += 1
                self.runtime.request.clear()
                self.runtime.chunks.stage(np.zeros((50, 14)), self.runtime.chunks.generation)
        return prefetch

    def test_execution_limits_count_full_14d_packets_and_discard_chunks(self):
        for limit in (1, 10, 20, 21):
            with self.subTest(limit=limit):
                generation = self.arm_execution_limit(limit)
                before = len(self.runtime.pubs["approved_action"].messages)
                prefetch = self.run_execution_ticks(limit + 3)
                packets = self.runtime.pubs["approved_action"].messages[before:]
                self.assertEqual(len(packets), limit)
                self.assertTrue(all(len(packet.data) == 14 for packet in packets))
                self.assertEqual(self.runtime.execution_steps, limit)
                self.assertEqual(prefetch, 0 if limit <= 10 else 1)
                self.assertFalse(self.runtime.chunks.enabled)
                self.assertFalse(self.runtime.chunks.active)
                self.assertIsNone(self.runtime.chunks.pending)
                self.assertFalse(self.runtime.request.is_set())
                self.assertFalse(self.runtime.chunks.stage(np.zeros((50, 14)), generation))
                self.assertEqual(self.runtime.fault_reason, "max_execution_steps_reached")
                self.assertTrue(self.runtime.pubs["hold"].messages[-1].data)
                self.assertEqual(self.runtime.pubs["safety_state"].messages[-1].data, "HOLD")
                last = json.loads(self.runtime.pubs["execution_step"].messages[-1].data)
                self.assertEqual(last["execution_steps"], limit)

    def test_unlimited_default_and_zero(self):
        self.assertEqual(self.runtime.max_execution_steps, 0)
        self.arm_execution_limit(0)
        self.run_execution_ticks(45)
        self.assertEqual(self.runtime.execution_steps, 45)
        self.assertTrue(self.runtime.chunks.enabled)

    def test_limit_stop_or_can_fault_midway_and_restart_resets_counter(self):
        for fault in (False, True):
            generation = self.arm_execution_limit(10)
            self.run_execution_ticks(4)
            if fault:
                self.runtime._controller_fault(String(data="right: CAN feedback stalled"))
            else:
                self.runtime.stop(None)
            before = len(self.runtime.pubs["approved_action"].messages)
            self.run_execution_ticks(12)
            self.assertEqual(len(self.runtime.pubs["approved_action"].messages), before)
            self.assertEqual(self.runtime.execution_steps, 4)
            self.assertFalse(self.runtime.chunks.stage(np.zeros((50, 14)), generation))
            self.arm_execution_limit(10)
            self.assertEqual(self.runtime.execution_steps, 0)
            self.run_execution_ticks(12)
            self.assertEqual(self.runtime.execution_steps, 10)

    def test_runtime_dry_limit_does_not_count_selection_or_release_hold(self):
        self.runtime.max_execution_steps = 10
        generation = self.start()
        self.runtime.chunks.stage(np.zeros((50,14)), generation)
        self.runtime.request.clear()
        self.run_execution_ticks(30)
        self.assertEqual(self.runtime.execution_steps, 0)
        self.assertTrue(self.runtime.chunks.enabled)
        self.assertEqual(self.runtime.pubs["approved_action"].messages, [])
        self.assertTrue(all(m.data for m in self.runtime.pubs["hold"].messages))
        self.runtime.stop(None)

    def test_publish_failure_not_counted_but_recorder_failure_after_send_is(self):
        self.arm_execution_limit(10)
        with mock.patch.object(self.runtime.pubs["approved_action"], "publish", side_effect=RuntimeError("send failed")):
            self.runtime.tick()
        self.assertEqual(self.runtime.execution_steps, 0)
        self.assertFalse(self.runtime.chunks.enabled)
        self.arm_execution_limit(10)
        def record(topic, message):
            if topic.endswith("/approved_action"):
                raise OSError("record failed after publish")
        with mock.patch.object(self.runtime, "_record", side_effect=record):
            self.runtime.tick()
        self.assertEqual(self.runtime.execution_steps, 1)
        self.assertFalse(self.runtime.chunks.enabled)

    def test_limit_metadata_records_actual_send_count_and_reason(self):
        self.runtime.cfg["recording"].update(enabled=True, root=str(self.path / "logs"))
        self.arm_execution_limit(10)
        self.run_execution_ticks(12)
        self.runtime.stop(None)
        saved = json.loads(next((self.path / "logs").glob("*/*_metadata.json")).read_text())
        self.assertEqual(saved["execution_steps"], 10)
        self.assertEqual(saved["max_execution_steps"], 10)
        self.assertEqual(saved["fault_reason"], "max_execution_steps_reached")

    def test_limit_discards_actual_inflight_prefetch_response(self):
        self.arm_execution_limit(20)
        entered, release = threading.Event(), threading.Event()
        def infer(observation):
            entered.set()
            release.wait(2)
            return {"actions": np.zeros((50,14))}
        self.runtime.client = types.SimpleNamespace(infer=infer, close=lambda: None, last_timing_ms={})
        worker = threading.Thread(target=self.runtime._inference_loop)
        worker.start()
        try:
            for _ in range(16):
                self.runtime.tick()
            self.assertTrue(entered.wait(1))
            for _ in range(4):
                self.runtime.tick()
            self.assertEqual(self.runtime.execution_steps,20)
            self.assertFalse(self.runtime.chunks.enabled)
            release.set()
        finally:
            release.set()
            self.runtime.shutdown_event.set()
            worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertIsNone(self.runtime.chunks.pending)
        self.assertEqual(self.runtime.pubs["policy_chunk"].messages,[])
        self.runtime.tick()
        self.assertEqual(self.runtime.execution_steps,20)

    def test_execution_limit_config_rejects_nonintegers(self):
        path = self.path / "runtime.yaml"
        config = yaml.safe_load(path.read_text())
        for value in (-1, True, 10.0, "10", None):
            config["runtime"]["max_execution_steps"] = value
            path.write_text(yaml.safe_dump(config))
            with mock.patch.dict(os.environ, RUNTIME_CONFIG=str(path)):
                with self.assertRaisesRegex(ValueError, "max_execution_steps"):
                    Runtime()
        config["runtime"].pop("max_execution_steps")
        path.write_text(yaml.safe_dump(config))
        with mock.patch.dict(os.environ, RUNTIME_CONFIG=str(path)):
            legacy = Runtime()
        self.assertEqual(legacy.max_execution_steps, 0)
        legacy.client.close()

    def test_inference_error_initial_timeout_and_recorder_failure(self):
        for fault in ("recorder", "initial_timeout"):
            with self.subTest(fault=fault):
                self.start()
                if fault == "inference_timeout":
                    self.runtime.inflight_generation = self.runtime.chunks.generation
                    self.runtime.inflight_since = time.monotonic() - 10
                elif fault == "recorder":
                    self.runtime.recorder = types.SimpleNamespace(error="disk full", record=lambda *a: False, metadata={})
                else:
                    self.runtime.started = time.monotonic() - 10
                self.runtime.tick()
                self.assertFalse(self.runtime.chunks.enabled)
                self.runtime.recorder = None

    def test_inference_thread_error_and_delayed_response(self):
        generation = self.start()
        entered, release = threading.Event(), threading.Event()

        def infer(_obs):
            entered.set()
            release.wait(1)
            return {"actions": np.zeros((50, 14))}

        self.runtime.client = types.SimpleNamespace(infer=infer, close=lambda: None, last_timing_ms={})
        thread = threading.Thread(target=self.runtime._inference_loop)
        thread.start()
        self.assertTrue(entered.wait(1))
        self.runtime.stop(None)
        release.set()
        self.runtime.shutdown_event.set()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertIsNone(self.runtime.chunks.pending)
        self.assertEqual(self.runtime.pubs["raw_action"].messages, [])

    def test_bad_policy_output_stops_inference_session(self):
        self.start()
        self.runtime.client = types.SimpleNamespace(
            infer=lambda obs: {"actions": np.zeros((50, 7))},
            close=lambda: None, last_timing_ms={})
        thread = threading.Thread(target=self.runtime._inference_loop)
        thread.start()
        deadline = time.monotonic() + 1
        while self.runtime.chunks.enabled and time.monotonic() < deadline:
            time.sleep(.005)
        self.runtime.shutdown_event.set()
        thread.join(1)
        self.assertFalse(self.runtime.chunks.enabled)
        self.assertIsNone(self.runtime.chunks.pending)
        self.assertTrue(self.runtime.pubs["hold"].messages[-1].data)
        self.assertIn("inference", self.runtime.fault_reason)

    def fail_one_request(self, error=None, delay=0):
        from piper_vla_common.policy_client import PolicyTimeout
        def infer(observation):
            if delay:
                time.sleep(delay)
            self.runtime.shutdown_event.set()
            if error:
                raise error
            return {"actions": np.zeros((50, 14))}
        self.runtime.client = types.SimpleNamespace(infer=infer, close=mock.Mock(), last_timing_ms={})
        self.runtime.request.set()
        worker = threading.Thread(target=self.runtime._inference_loop)
        worker.start()
        worker.join(1)
        self.assertFalse(worker.is_alive())

    def test_communication_timeout_keeps_active_chunk_and_token_until_underrun(self):
        from piper_vla_common.policy_client import PolicyTimeout
        self.arm_execution_limit(0)
        self.runtime.tick()  # Activate and execute action zero.
        token = self.runtime.controller_token
        generation = self.runtime.chunks.generation
        self.fail_one_request(PolicyTimeout("receive timeout"))
        self.assertTrue(self.runtime.chunks.enabled)
        self.assertEqual(self.runtime.chunks.generation, generation)
        self.assertEqual(self.runtime.controller_token, token)
        self.assertIsNone(self.runtime.fault_reason)
        self.assertIn("timeout", self.runtime.last_communication_error)
        self.assertFalse(self.runtime.request.is_set())  # No automatic retry.
        for _ in range(19):
            self.runtime.tick()
        self.assertEqual(self.runtime.execution_steps, 20)
        self.assertTrue(self.runtime.chunks.enabled)
        self.runtime.tick()
        self.assertEqual(self.runtime.fault_reason, "chunk_underrun")
        self.assertIsNone(self.runtime.controller_token)

    def test_deadline_rechecked_without_a_watchdog_tick(self):
        self.arm_execution_limit(0)
        self.runtime.tick()
        self.runtime.cfg["runtime"]["inference_timeout_s"] = .01
        self.fail_one_request(delay=.025)
        self.assertIsNone(self.runtime.chunks.pending)
        self.assertEqual(self.runtime.pubs["policy_chunk"].messages, [])
        self.assertTrue(self.runtime.chunks.enabled)
        self.assertIn("deadline", self.runtime.last_communication_error)
        self.runtime.tick()
        self.assertEqual(self.runtime.execution_steps, 2)

    def test_watchdog_timeout_records_but_does_not_fault_valid_chunk(self):
        self.arm_execution_limit(0)
        token = self.runtime.controller_token
        self.runtime.inflight_generation = self.runtime.chunks.generation
        self.runtime.inflight_since = time.monotonic() - 10
        self.runtime.tick()
        self.assertTrue(self.runtime.chunks.enabled)
        self.assertEqual(self.runtime.execution_steps, 1)
        self.assertEqual(self.runtime.controller_token, token)
        self.assertEqual(self.runtime.last_communication_error, "policy_timeout")

    def test_communication_error_does_not_discard_pending_chunk(self):
        from piper_vla_common.policy_client import PolicyCommunicationError
        self.arm_execution_limit(0)
        self.runtime.tick()
        self.runtime.chunks.stage(np.ones((50, 14)), self.runtime.chunks.generation)
        self.fail_one_request(PolicyCommunicationError("disconnected"))
        self.assertIsNotNone(self.runtime.chunks.pending)
        for _ in range(20):
            self.runtime.tick()
        self.assertEqual(self.runtime.execution_steps, 21)
        np.testing.assert_equal(self.runtime.pubs["approved_action"].messages[-1].data, np.ones(14))

    def test_initial_communication_error_waits_without_action_then_startup_hold(self):
        from piper_vla_common.policy_client import PolicyTimeout
        self.runtime.dry = False
        self.start()
        self.fail_one_request(PolicyTimeout("first request"))
        self.runtime.tick()
        self.assertTrue(self.runtime.chunks.enabled)
        self.assertIsNone(self.runtime.controller_started)
        self.assertEqual(self.runtime.pubs["approved_action"].messages, [])
        self.assertFalse(self.runtime.request.is_set())
        self.runtime.started -= 10
        self.runtime.tick()
        self.assertEqual(self.runtime.fault_reason, "initial_chunk_timeout")
        self.assertIsNone(self.runtime.controller_token)

    def test_noncommunication_oserror_during_publication_is_a_safety_fault(self):
        self.arm_execution_limit(0)
        self.runtime.tick()
        original = self.runtime._publish
        def publish(name, message):
            if name == "state":
                raise OSError("recording failure")
            return original(name, message)
        with mock.patch.object(self.runtime, "_publish", side_effect=publish):
            self.fail_one_request()
        self.assertFalse(self.runtime.chunks.enabled)
        self.assertIsNone(self.runtime.controller_token)
        self.assertIn("recording failure", self.runtime.fault_reason)

    def test_expired_or_invalid_controller_token_overrides_chunk_continuation(self):
        for invalid in ("expired", "generation", "missing"):
            self.arm_execution_limit(0)
            self.runtime.tick()
            before = self.runtime.execution_steps
            self.runtime._communication_error("websocket failure")
            if invalid == "expired":
                self.runtime.controller_token_expires = time.monotonic() - 1
            elif invalid == "generation":
                self.runtime.controller_token_generation -= 1
            else:
                self.runtime.controller_token = None
            self.runtime.tick()
            self.assertFalse(self.runtime.chunks.enabled)
            self.assertEqual(self.runtime.execution_steps, before)
            self.assertIsNone(self.runtime.controller_token)
            self.assertIn("controller_token", self.runtime.fault_reason)

    def test_dry_run_prefetch_advances_and_stop_is_latched(self):
        generation = self.start()
        self.runtime.request.clear()
        self.runtime.chunks.stage(np.zeros((50, 14)), generation)
        for _ in range(self.runtime.chunks.prefetch + 1):
            self.runtime.tick()
        self.assertTrue(self.runtime.request.is_set())
        self.runtime.stop(None)
        count = len(self.runtime.pubs["selected_action"].messages)
        self.runtime.tick()
        self.assertEqual(len(self.runtime.pubs["selected_action"].messages), count)
        self.assertTrue(all(item.data for item in self.runtime.pubs["hold"].messages))
        self.assertEqual(self.runtime.pubs["approved_action"].messages, [])

    def test_recording_enabled_dry_run_session(self):
        self.runtime.cfg["recording"].update(enabled=True, root=str(self.path / "logs"))
        self.start()
        self.runtime.stop(None)
        metadata = list((self.path / "logs").glob("*/*_metadata.json"))
        self.assertEqual(len(metadata), 1)
        saved = json.loads(metadata[0].read_text())
        self.assertTrue(saved["runtime_dry_run"])
        self.assertEqual(saved["fault_reason"], "stop_service")


class RecorderTests(unittest.TestCase):
    def test_mcap_npz_metadata_roundtrip(self):
        with tempfile.TemporaryDirectory() as root:
            recorder = RuntimeRecorder(root, {"contract": POLICY_CONTRACT}, 32)
            recorder.start()
            actions = np.arange(700).reshape(50, 14)
            recorder.record("/vla_dual/policy_chunk", Float64MultiArray(data=actions.reshape(-1).tolist()))
            recorder.record("/vla_dual/state", Float64MultiArray(data=list(range(14))))
            recorder.record("/vla_dual/left/target_joint_command", state())
            for name in ("front", "left_wrist", "right_wrist"):
                recorder.record("/camera/" + name + "/compressed", CompressedImage())
            recorder.record("/vla_dual/safety_reason", String(data="test stop"))
            recorder.close("test")
            data = np.load(next(recorder.run_dir.glob("*.npz")))
            np.testing.assert_equal(data["vla_dual__policy_chunk"][0], actions)
            self.assertEqual(data["vla_dual__state"].shape, (1, 14))
            self.assertEqual(data["vla_dual__right__target_joint_command"].shape, (0, 7))
            with next(recorder.run_dir.glob("*.mcap")).open("rb") as stream:
                reader = make_reader(stream, decoder_factories=[DecoderFactory()])
                messages = list(reader.iter_decoded_messages())
            self.assertEqual(len(messages), 7)
            metadata = json.loads(next(recorder.run_dir.glob("*_metadata.json")).read_text())
            self.assertEqual(metadata["stop_reason"], "test")
            self.assertIsNone(metadata["recorder_error"])

    def test_recorder_failure_is_visible(self):
        with tempfile.TemporaryDirectory() as root:
            recorder = RuntimeRecorder(root, {}, 1)
            recorder.start()
            recorder.record("/vla_dual/state", Float64MultiArray(data=[1]))
            self.assertIn("invalid", recorder.error)
            recorder.close("failure")

    def test_configured_state_topics_remain_in_npz(self):
        with tempfile.TemporaryDirectory() as root:
            recorder = RuntimeRecorder(root, {}, vector_topics={"/custom/left_state": 7})
            recorder.start()
            recorder.record("/custom/left_state", state())
            recorder.close("test")
            data = np.load(next(recorder.run_dir.glob("*.npz")))
            self.assertEqual(data["custom__left_state"].shape, (1, 7))


class TransformTests(unittest.TestCase):
    def test_real_openpi_quantile_delta_roundtrip_32_to_14(self):
        # Execute the *unchanged vendored* numeric transform classes directly,
        # avoiding GPU/JAX/Flax imports. Tree helpers operate on flat test dicts.
        source = ROOT / "vendor/openpi_pbl5/src/openpi/transforms.py"
        names = {"Normalize", "Unnormalize", "DeltaActions", "AbsoluteActions", "apply_tree", "pad_to_dim", "_assert_quantile_stats"}
        tree = ast.parse(source.read_text())
        selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
        selected.insert(0, ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0))
        scope = {"np": np, "dataclasses": dataclasses, "DataTransformFn": object,
                 "flatten_dict": lambda data: dict(data), "unflatten_dict": lambda data: dict(data)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), str(source), "exec"), scope)
        current = np.array([.2] * 6 + [.04] + [.4] * 6 + [.06])
        absolute = np.tile(current + np.array([.1] * 6 + [0] + [.2] * 6 + [0]), (50, 1))
        delta = scope["DeltaActions"](DELTA_MASK)({"state": current.copy(), "actions": absolute.copy()})
        np.testing.assert_allclose(delta["actions"][0, [0, 7, 6, 13]], [.1, .2, .04, .06])
        stats = {key: types.SimpleNamespace(q01=np.full(14, -1.), q99=np.full(14, 1.)) for key in ("state", "actions")}
        normalized = scope["Normalize"](stats, use_quantiles=True)(delta)
        padded = {key: scope["pad_to_dim"](value, 32) for key, value in normalized.items()}
        restored = scope["Unnormalize"](stats, use_quantiles=True)(padded)
        restored = scope["AbsoluteActions"](DELTA_MASK)(restored)
        output = DualOutputs()(restored)["actions"]
        self.assertEqual(output.shape, (50, 14))
        np.testing.assert_allclose(output, absolute, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
