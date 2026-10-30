#!/usr/bin/env python3
"""Unit tests for prompt-scoped VLA runtime sessions."""

import numpy as np
import threading
import types
import unittest
from unittest import mock

from piper_vla_runtime import node as runtime_node


PROMPT_A = "Pick the object in area 1 and place it in the red area."
PROMPT_B = "Pick the object in area 2 and place it in the red area."


class PromptSessionTest(unittest.TestCase):
    def make_runtime(self):
        runtime = runtime_node.Runtime.__new__(runtime_node.Runtime)
        runtime.chunks = runtime_node.ChunkQueue(1, 0, runtime_node.ActionContract(7), reset_ids=False, underrun_error=False)
        runtime._service_lock = threading.Lock()
        runtime._session_lock = threading.RLock()
        runtime._chunk_lock = threading.Lock()
        runtime._stats_lock = threading.Lock()
        runtime.observations = runtime_node.ObservationStore(runtime_node.OBSERVATION_NAMES)
        runtime._observation_versions = runtime.observations.versions
        runtime._start_observation_versions = None
        runtime.state = types.SimpleNamespace(position=[0.0] * 7)
        runtime.front = np.zeros((224, 224, 3), np.uint8)
        runtime.wrist = np.zeros((224, 224, 3), np.uint8)
        runtime.ee = object()
        runtime.enabled = False
        runtime.current_prompt = None
        runtime.session_generation = 0
        runtime._awaiting_initial_chunk = False
        runtime._last_hold_reason = None
        runtime._inference_request = threading.Event()
        runtime.recorder = None
        runtime.client = object()
        runtime.prompt_pub = object()
        runtime.hold_pub = object()
        runtime.cfg = {
            "runtime": {"dry_run": True},
            "observation": {
                "state_key": "observation/state",
                "front_key": "observation/image",
                "wrist_key": "observation/wrist_image",
            },
            "robot": {"approved_action_topic": "/vla/approved_action"},
            "safety": {"z_min": 0.15},
        }
        runtime.is_pbl8 = False
        runtime.events = []
        runtime._recording_gate = lambda: (
            False, "runtime_dry_run=true", None
        )
        runtime._stop_recording = lambda _reason: None
        runtime._publish_recorded = lambda _publisher, topic, message: (
            runtime.events.append((topic, message.data))
        )
        runtime.policy_state = lambda state: np.asarray(state.position, np.float32)
        runtime.image = lambda image, rotate_left_90=False: image
        return runtime

    @staticmethod
    def request(prompt):
        return types.SimpleNamespace(prompt=prompt)

    @staticmethod
    def publish_fresh_observations(runtime):
        for name in runtime_node.OBSERVATION_NAMES:
            runtime.observations.update(name, getattr(runtime, name))

    def test_start_prompt_is_used_by_next_request(self):
        runtime = self.make_runtime()
        with mock.patch.object(runtime_node.rospy, "loginfo"):
            response = runtime.start(self.request(PROMPT_A))
        self.assertTrue(response.success)
        snapshot, reason = runtime._inference_session_snapshot()
        self.assertIsNone(snapshot)
        self.assertIn("post-start observations", reason)
        self.publish_fresh_observations(runtime)
        snapshot, reason = runtime._inference_session_snapshot()
        self.assertIsNone(reason)
        _, observation = runtime._build_policy_observation(snapshot)
        self.assertEqual(observation["prompt"], PROMPT_A)

    def test_stop_then_start_uses_new_prompt_and_same_client(self):
        runtime = self.make_runtime()
        client = runtime.client
        with mock.patch.object(runtime_node.rospy, "loginfo"):
            self.assertTrue(runtime.start(self.request(PROMPT_A)).success)
            self.assertTrue(runtime.stop(None).success)
            self.assertTrue(runtime.start(self.request(PROMPT_B)).success)
        self.publish_fresh_observations(runtime)
        snapshot, reason = runtime._inference_session_snapshot()
        self.assertIsNone(reason)
        _, observation = runtime._build_policy_observation(snapshot)
        self.assertEqual(observation["prompt"], PROMPT_B)
        self.assertIs(runtime.client, client)

    def test_delayed_response_from_old_generation_is_discarded(self):
        runtime = self.make_runtime()
        with mock.patch.object(runtime_node.rospy, "loginfo"):
            runtime.start(self.request(PROMPT_A))
            old_generation = runtime.session_generation
            runtime.stop(None)
            runtime.start(self.request(PROMPT_B))
            accepted = runtime._stage_chunk(
                [[1.0] * 7], old_generation
            )
        self.assertFalse(accepted)
        self.assertIsNone(runtime.chunks.pending)

    def test_start_while_running_is_rejected(self):
        runtime = self.make_runtime()
        with mock.patch.object(runtime_node.rospy, "loginfo"):
            self.assertTrue(runtime.start(self.request(PROMPT_A)).success)
            response = runtime.start(self.request(PROMPT_B))
        self.assertFalse(response.success)
        self.assertIn("already running", response.message)
        self.assertEqual(runtime.current_prompt, PROMPT_A)

    def test_empty_and_whitespace_prompts_are_rejected(self):
        runtime = self.make_runtime()
        for prompt in ("", "   ", "\n\t"):
            response = runtime.start(self.request(prompt))
            self.assertFalse(response.success)
            self.assertIn("must not be empty", response.message)
        self.assertFalse(runtime.enabled)

    def test_extremely_long_prompt_is_rejected(self):
        runtime = self.make_runtime()
        response = runtime.start(
            self.request("x" * (runtime_node.MAX_PROMPT_CHARACTERS + 1))
        )
        self.assertFalse(response.success)
        self.assertIn("too long", response.message)


if __name__ == "__main__":
    unittest.main()
