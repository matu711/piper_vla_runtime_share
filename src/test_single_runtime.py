"""Single-arm request acceptance checks using only the existing ROS doubles."""
import threading
import time
import types
import unittest
from unittest import mock

import test_prompt_sessions
from piper_vla_common.policy_client import StaleResponse


class SingleRequestTests(unittest.TestCase):
    def runtime(self):
        runtime = test_prompt_sessions.PromptSessionTest().make_runtime()
        runtime._stop_inference = threading.Event()
        runtime.start(types.SimpleNamespace(prompt="single test"))
        test_prompt_sessions.PromptSessionTest.publish_fresh_observations(runtime)
        return runtime

    def test_ee_is_acquired_but_never_sent_to_model(self):
        runtime = self.runtime()
        snapshot, reason = runtime._inference_session_snapshot()
        self.assertIsNone(reason)
        self.assertIn("ee", snapshot.messages)
        _, obs = runtime._build_policy_observation(snapshot)
        self.assertEqual(set(obs), {"observation/state", "observation/image", "observation/wrist_image", "prompt"})

    def test_expired_response_does_not_replace_valid_single_chunk_or_stop_session(self):
        runtime = self.runtime()
        runtime._new_inference_request(runtime._inference_session_snapshot()[0])
        runtime._stage_chunk([[.1] * 7], runtime.session_generation)
        def infer(obs):
            time.sleep(.015)
            runtime._stop_inference.set()
            return {"actions": [[.9] * 7]}
        runtime.client = types.SimpleNamespace(infer=infer, timeout=.005, close=lambda: None)
        runtime._inference_request.set()
        runtime._inference_loop()
        self.assertTrue(runtime.enabled)
        self.assertEqual(runtime.chunks.pending[1].tolist(), [[.1] * 7])

    def test_first_request_only_and_explicit_restart_reset_timeout(self):
        runtime = self.runtime()
        runtime.client = types.SimpleNamespace(timeout=2.)
        snapshot, _ = runtime._inference_session_snapshot()
        now = time.monotonic()
        with mock.patch("piper_vla_runtime.node.time.monotonic", return_value=now):
            first = runtime._new_inference_request(snapshot)
            self.assertEqual(first.deadline, now + 30.)
            # Still awaiting the first chunk: failure must not grant 30s again.
            self.assertTrue(runtime._awaiting_initial_chunk)
            self.assertEqual(runtime._new_inference_request(snapshot).deadline, now + 2.)
            runtime.stop(None)
            runtime.start(types.SimpleNamespace(prompt="restart"))
            self.assertIsNone(runtime._inference_session_snapshot()[0])
            test_prompt_sessions.PromptSessionTest.publish_fresh_observations(runtime)
            with self.assertRaises(StaleResponse):
                runtime._new_inference_request(snapshot)
            snapshot, _ = runtime._inference_session_snapshot()
            self.assertEqual(runtime._new_inference_request(snapshot).deadline, now + 30.)
            with self.assertRaises(StaleResponse):
                first.check()

    def test_first_response_after_deadline_is_discarded(self):
        runtime = self.runtime()
        runtime.raw_pub = runtime.type_pub = object()
        runtime.cfg["runtime"]["initial_inference_timeout_s"] = .005
        def infer(obs):
            time.sleep(.015)
            runtime._stop_inference.set()
            return {"actions": [[.9] * 7]}
        runtime.client = types.SimpleNamespace(infer=infer, timeout=2., close=mock.Mock())
        runtime._inference_loop()
        self.assertIsNone(runtime.chunks.pending)
        self.assertTrue(runtime._awaiting_initial_chunk)
        self.assertFalse(runtime._inference_request.is_set())
        self.assertFalse(any(topic == "/vla/raw_action" for topic, _ in runtime.events))
        runtime.client.close.assert_called()

    def test_initial_failure_waits_without_automatic_retry(self):
        runtime = self.runtime()
        def infer(obs):
            runtime._stop_inference.set()
            raise TimeoutError("no first chunk")
        runtime.client = types.SimpleNamespace(infer=infer, timeout=2., close=lambda: None)
        runtime._inference_request.set()
        runtime._inference_loop()
        self.assertTrue(runtime.enabled)
        self.assertTrue(runtime._awaiting_initial_chunk)
        self.assertIsNone(runtime.chunks.pending)
        self.assertFalse(runtime._inference_request.is_set())

    def test_success_keeps_raw_action_and_chunk_values_without_state_addition(self):
        runtime = self.runtime()
        runtime.raw_pub = runtime.type_pub = object()
        runtime.cfg["policy"] = {"action_type": "joint_absolute"}
        runtime.execution_chunk_length = 1
        timings = []
        runtime._record_chunk_inference = lambda *args: timings.append(args)
        def infer(obs):
            runtime._stop_inference.set()
            return {"actions": [[.2] * 6 + [.04]]}
        runtime.client = types.SimpleNamespace(infer=infer, timeout=2., last_timing_ms={})
        runtime._inference_request.set()
        runtime._inference_loop()
        raw = next(data for topic, data in runtime.events if topic == "/vla/raw_action")
        self.assertEqual(raw, [.2] * 6 + [.04])
        self.assertEqual(runtime.chunks.pending[1].tolist(), [raw])
        self.assertEqual(len(timings), 1)
