#!/usr/bin/env python3
"""Policy client: observes ROS, records raw actions, gates only target EEF Z."""
import copy
import math
import os
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import rospy
import yaml
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Bool, Float64, Float64MultiArray, String
from std_srvs.srv import Trigger, TriggerResponse
from piper_vla_runtime_msgs.srv import StartInference, StartInferenceResponse
from piper_vla_common.policy_client import (OpenPIClient, InferenceRequest, infer_request,
                                          PolicyCommunicationError, StaleResponse)
from piper_vla_common.execution import ActionContract, ChunkQueue
from piper_vla_common.observations import ObservationStore, decode_rgb
from piper_vla_common.adapters import prepare_image
from piper_vla_runtime.observation_adapter import policy_state as adapt_single_state, build_observation
from piper_vla_runtime.pbl8_gripper import (
    PBL8_GRIPPER_RUNTIME_MAX,
    PBL8_GRIPPER_RUNTIME_MIN,
    apply_pbl8_gripper_saturation,
    is_within_pbl8_gripper_saturation_margin,
)
from piper_vla_runtime.runtime_recorder import (
    RuntimeRecorder,
    TOPIC_END_POSE,
    TOPIC_FRONT_CAMERA,
    TOPIC_JOINT_STATE,
    TOPIC_WRIST_CAMERA,
)

PBL8_ACTION_TYPES = {
    # Legacy bundle name.
    "ee_delta_cumulative_reference",
    # Canonical name used by piper_pi0_pbl8_lora_step99999.
    "ee_delta_physical_world_per_control_step",
}
PIPER_CONTROLLER_DRY_RUN_PARAM = "/vla_piper_controller/dry_run"
MAX_PROMPT_CHARACTERS = 2048
OBSERVATION_NAMES = ("state", "front", "wrist", "ee")
ACTION_TYPE_ALIASES = {
    # This manifest name describes model space. The checkpoint's registered
    # output transform applies AbsoluteActions to joints 1..6 before the
    # WebSocket response, so the robot-facing contract is joint_absolute.
    "joint_delta_first_6_plus_absolute_gripper": "joint_absolute",
    "joint_delta_first_6_with_absolute_gripper": "joint_absolute",
    "absolute_joint_position_after_output_transform": "joint_absolute",
}
class Runtime:
    enabled = property(lambda self: self.chunks.enabled,
                       lambda self, value: setattr(self.chunks, "enabled", value))
    session_generation = property(lambda self: self.chunks.generation,
                                  lambda self, value: setattr(self.chunks, "generation", value))
    current_prompt = property(lambda self: self.chunks.session.prompt,
                              lambda self, value: setattr(self.chunks.session, "prompt", value))
    _awaiting_initial_chunk = property(lambda self: self.chunks.awaiting_initial,
                                      lambda self, value: setattr(self.chunks, "awaiting_initial", value))

    def __init__(self):
        cfg_path = os.environ.get("RUNTIME_CONFIG", "/runtime/config/runtime.yaml")
        self.cfg = yaml.safe_load(Path(cfg_path).read_text())
        manifest = Path(self.cfg["model"]["manifest"])
        self.manifest = yaml.safe_load(manifest.read_text()) if manifest.exists() else {}
        self.manifest = self.manifest or {}
        self.manifest_action_type = self.manifest.get(
            "action_type", self.cfg["policy"]["action_type"]
        )
        self.cfg["policy"]["action_type"] = ACTION_TYPE_ALIASES.get(
            self.manifest_action_type, self.manifest_action_type
        )
        self.cfg["runtime"]["prompt"] = self.manifest.get("task", {}).get("default_prompt", self.cfg["runtime"]["prompt"])
        self.is_pbl8 = self.cfg["policy"]["action_type"] in PBL8_ACTION_TYPES
        self._service_lock = threading.Lock()
        self._session_lock = threading.RLock()
        self.observations = ObservationStore(OBSERVATION_NAMES)
        self._observation_versions = self.observations.versions
        self._start_observation_versions = None
        self.state = self.front = self.wrist = self.ee = None
        self.chunks = ChunkQueue(1, 0, ActionContract(7), reset_ids=False, underrun_error=False)
        self.enabled = False
        self.current_prompt = None
        self.session_generation = 0
        self.model_action_horizon = int(
            self.manifest.get("action_horizon", self.cfg["policy"].get("action_horizon", 50))
        )
        self._chunk_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        # True only between ~start and activation of the first inferred chunk.
        # The control loop deliberately waits in this state instead of treating
        # normal initial inference latency as a chunk underrun.
        self._awaiting_initial_chunk = False
        self._initial_inference_pending = False
        self._stop_inference = threading.Event()
        self._inference_request = threading.Event()
        self._last_hold_reason = None
        self.recorder = None
        rospy.init_node("piper_vla_runtime")
        if self.manifest_action_type != self.cfg["policy"]["action_type"]:
            rospy.loginfo(
                "Policy action contract resolved: manifest=%s WebSocket/controller=%s",
                self.manifest_action_type,
                self.cfg["policy"]["action_type"],
            )
        rospy.loginfo(
            "Recording gate: enabled only when runtime dry_run=false and %s=false",
            PIPER_CONTROLLER_DRY_RUN_PARAM,
        )
        self._configure_pbl8_safety()
        observation, robot = self.cfg["observation"], self.cfg["robot"]
        rospy.Subscriber(observation["state_topic"], JointState,
                         self._set("state", TOPIC_JOINT_STATE), queue_size=1)
        rospy.Subscriber(observation["front_camera_topic"], CompressedImage,
                         self._set("front", TOPIC_FRONT_CAMERA), queue_size=1)
        rospy.Subscriber(observation["wrist_camera_topic"], CompressedImage,
                         self._set("wrist", TOPIC_WRIST_CAMERA), queue_size=1)
        rospy.Subscriber(observation["ee_pose_topic"], PoseStamped,
                         self._set("ee", TOPIC_END_POSE), queue_size=1)
        self.raw_pub = rospy.Publisher("/vla/raw_action", Float64MultiArray, queue_size=10)
        # /vla/raw_action keeps its existing meaning: actions[0] for each
        # policy response. This topic contains the action selected at control rate.
        self.selected_pub = rospy.Publisher("/vla/selected_action", Float64MultiArray, queue_size=10)
        self.type_pub = rospy.Publisher("/vla/action_type", String, queue_size=10, latch=True)
        self.target_pub = rospy.Publisher("/vla/target_ee_pose", PoseStamped, queue_size=10)
        self.safety_pub = rospy.Publisher("/safety/state", String, queue_size=10)
        self.reason_pub = rospy.Publisher("/safety/reason", String, queue_size=10)
        self.z_pub = rospy.Publisher("/safety/target_z", Float64, queue_size=10)
        self.command_pub = rospy.Publisher(robot["approved_action_topic"], Float64MultiArray, queue_size=1)
        self.hold_pub = rospy.Publisher(robot["hold_topic"], Bool, queue_size=1, latch=True)
        self.prompt_pub = rospy.Publisher("/vla/current_prompt", String, queue_size=1, latch=True)
        rospy.Service("~start", StartInference, self.start)
        rospy.Service("~stop", Trigger, self.stop)
        host = os.environ.get("POLICY_HOST", self.cfg["policy"]["host"])
        port = int(os.environ.get("POLICY_PORT", self.cfg["policy"]["port"]))
        # Hard request bound is separate from the inference-rate statistics target.
        timeout = float(self.cfg["runtime"].get("inference_timeout_s", 2.0))
        initial_timeout = float(self.cfg["runtime"].get("initial_inference_timeout_s", 30.0))
        if not math.isfinite(initial_timeout) or initial_timeout <= 0:
            raise ValueError("initial_inference_timeout_s must be finite and positive")
        self.client = OpenPIClient(host, port, timeout, action_contract=ActionContract(7))
        runtime_cfg = self.cfg["runtime"]
        self.inference_rate_hz = float(
            rospy.get_param(
                "~inference_rate_hz",
                runtime_cfg.get("inference_rate_hz", runtime_cfg.get("rate_hz", 5.0)),
            )
        )
        if self.inference_rate_hz <= 0.0:
            raise ValueError("inference_rate_hz must be positive")
        self.control_rate_hz = float(rospy.get_param(
            "~control_rate_hz", runtime_cfg.get("control_rate_hz", 20.0)
        ))
        self.execution_chunk_length = int(rospy.get_param(
            "~execution_chunk_length", runtime_cfg.get("execution_chunk_length", 10)
        ))
        self.prefetch_start_index = int(rospy.get_param(
            "~prefetch_start_index", runtime_cfg.get("prefetch_start_index", 7)
        ))
        if self.control_rate_hz <= 0.0:
            raise ValueError("control_rate_hz must be positive")
        if not 1 <= self.execution_chunk_length <= self.model_action_horizon:
            raise ValueError(
                "execution_chunk_length must satisfy 1 <= length <= "
                f"model action_horizon ({self.model_action_horizon}), got {self.execution_chunk_length}"
            )
        if not 0 <= self.prefetch_start_index < self.execution_chunk_length:
            raise ValueError(
                "prefetch_start_index must satisfy 0 <= index < execution_chunk_length "
                f"({self.execution_chunk_length}), got {self.prefetch_start_index}"
            )
        self.client.minimum_actions = self.client.finite_rows = self.execution_chunk_length
        self.chunks.length, self.chunks.prefetch = self.execution_chunk_length, self.prefetch_start_index
        self.deadline_s = 1.0 / self.inference_rate_hz
        self.control_deadline_s = 1.0 / self.control_rate_hz
        self.statistics_period_s = float(runtime_cfg.get("statistics_period_s", 5.0))
        self._stats_started = time.perf_counter()
        self._last_statistics_log = self._stats_started
        self._chunk_stats = {
            "policy_count": 0,
            "policy_deadline_miss": 0,
            "command_count": 0,
            "control_deadline_miss": 0,
            "chunk_underrun_count": 0,
            "safety_hold_count": 0,
            "chunk_count": 0,
            "prefetch_count": 0,
            "observation_ms": deque(maxlen=500),
            "ws_send_ms": deque(maxlen=500),
            "response_wait_ms": deque(maxlen=500),
            "policy_infer_ms": deque(maxlen=500),
            "round_trip_ms": deque(maxlen=500),
            "control_period_ms": deque(maxlen=500),
        }
        rospy.loginfo(
            "READY: policy=%s:%s action=%s z_min=%.3f dry_run=%s; "
            "model_action_horizon=%d execution_chunk_length=%d prefetch_start_index=%d "
            "control_target=%.1f Hz; policy_deadline=%.1f ms; "
            "call /piper_vla_runtime/start",
            host, port, self.kind, self.zmin, self.dry,
            self.model_action_horizon, self.execution_chunk_length, self.prefetch_start_index,
            self.control_rate_hz, self.deadline_s * 1000.0,
        )

    def _configure_pbl8_safety(self):
        """Use controller ROS params as the shared PBL8 safety source."""
        pbl8_cfg = self.cfg.get("pbl8", {})
        translation_fallback = float(
            pbl8_cfg.get("max_translation_delta_m", 0.003)
        )
        rotation_fallback = float(pbl8_cfg.get("max_rotation_delta_rad", 0.01))
        controller_translation_param = (
            "/vla_piper_controller/pbl8_max_translation_delta_m"
        )
        controller_rotation_param = (
            "/vla_piper_controller/pbl8_max_rotation_delta_rad"
        )
        if rospy.has_param(controller_translation_param):
            self.pbl8_max_translation_delta_m = float(
                rospy.get_param(controller_translation_param)
            )
            translation_source = controller_translation_param
        else:
            self.pbl8_max_translation_delta_m = float(rospy.get_param(
                "~pbl8_max_translation_delta_m", translation_fallback
            ))
            translation_source = "runtime config/private param fallback"
        if rospy.has_param(controller_rotation_param):
            self.pbl8_max_rotation_delta_rad = float(
                rospy.get_param(controller_rotation_param)
            )
            rotation_source = controller_rotation_param
        else:
            self.pbl8_max_rotation_delta_rad = float(rospy.get_param(
                "~pbl8_max_rotation_delta_rad", rotation_fallback
            ))
            rotation_source = "runtime config/private param fallback"
        for name, value in (
            ("max_translation_delta_m", self.pbl8_max_translation_delta_m),
            ("max_rotation_delta_rad", self.pbl8_max_rotation_delta_rad),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("PBL8 %s must be finite and > 0" % name)
        if self.is_pbl8:
            rospy.loginfo(
                "PBL8 safety:\n"
                "  max_translation_delta_m=%.6f (source=%s)\n"
                "  max_rotation_delta_rad=%.6f (source=%s)",
                self.pbl8_max_translation_delta_m, translation_source,
                self.pbl8_max_rotation_delta_rad, rotation_source,
            )

    def _set(self, name, record_topic=None):
        def callback(message):
            with self._session_lock:
                setattr(self, name, message)
                self.observations.update(name, message)
            if record_topic is not None:
                self._record(record_topic, message)
        return callback

    @property
    def kind(self): return self.cfg["policy"]["action_type"]
    @property
    def zmin(self): return float(self.cfg["safety"]["z_min"])
    @property
    def dry(self): return bool(self.cfg["runtime"]["dry_run"])

    def _record(self, topic, message):
        recorder = self.recorder
        if recorder is not None:
            recorder.record(topic, message)

    def _publish_recorded(self, publisher, topic, message):
        publisher.publish(message)
        self._record(topic, message)

    def _recording_gate(self):
        """Return whether this run can move hardware and should be recorded."""
        if self.dry:
            return False, "runtime_dry_run=true", None
        if not rospy.has_param(PIPER_CONTROLLER_DRY_RUN_PARAM):
            return False, "piper_controller_dry_run_unavailable", None
        controller_dry_run = rospy.get_param(PIPER_CONTROLLER_DRY_RUN_PARAM)
        if not isinstance(controller_dry_run, bool):
            return False, "piper_controller_dry_run_not_bool", controller_dry_run
        if controller_dry_run:
            return False, "piper_controller_dry_run=true", True
        return True, "live_hardware_run", False

    def _recording_metadata(self, controller_dry_run, prompt, generation):
        recording_cfg = self.cfg.get("recording", {})
        return {
            "checkpoint": self.cfg["model"]["checkpoint"],
            "config": self.manifest.get("config_name"),
            "action_type": self.kind,
            "model_action_horizon": self.model_action_horizon,
            "execution_chunk_length": self.execution_chunk_length,
            "prefetch_start_index": self.prefetch_start_index,
            "control_rate_hz": self.control_rate_hz,
            "z_min": self.zmin,
            "gripper_range_m": recording_cfg.get("gripper_range_m", [0.0, 0.10]),
            "gripper_saturation_margin_m": recording_cfg.get(
                "gripper_saturation_margin_m", 0.01
            ),
            "runtime_dry_run": self.dry,
            "piper_controller_dry_run": controller_dry_run,
            "prompt": prompt,
            "session_generation": generation,
        }

    def _start_recording(self, controller_dry_run, prompt, generation):
        if self.recorder is not None:
            raise RuntimeError("recording is already active; call ~stop first")
        recording_cfg = self.cfg.get("recording", {})
        recorder = RuntimeRecorder(
            root=recording_cfg.get("root", self.cfg["logging"]["root"]),
            metadata=self._recording_metadata(
                controller_dry_run, prompt, generation
            ),
            queue_size=recording_cfg.get("queue_size", 512),
        )
        try:
            recorder.start()
        except Exception:
            # If setup failed after opening the file, make a best effort to
            # finish it before reporting start failure to the service caller.
            try:
                recorder.close(reason="start_failed")
            except Exception:
                pass
            raise
        self.recorder = recorder
        rospy.loginfo(
            "Recording started: run_dir=%s mcap=%s npz=%s",
            recorder.run_dir, recorder.mcap_path, recorder.npz_path,
        )

    def _stop_recording(self, reason):
        recorder, self.recorder = self.recorder, None
        if recorder is None:
            return
        try:
            result = recorder.close(reason=reason)
        except Exception as exc:
            rospy.logerr("Recording close failed: %s", exc)
            return
        rospy.loginfo(
            "Recording stopped: duration=%.3fs mcap=%s npz=%s samples=%d",
            result["duration_s"], result["mcap"], result["npz"], result["samples"],
        )

    @staticmethod
    def _validate_prompt(value):
        prompt = str(value).strip()
        if not prompt:
            raise ValueError("prompt must not be empty or whitespace only")
        if len(prompt) > MAX_PROMPT_CHARACTERS:
            raise ValueError(
                "prompt is too long: %d characters (maximum %d)" %
                (len(prompt), MAX_PROMPT_CHARACTERS)
            )
        return prompt

    def start(self, request):
        try:
            prompt = self._validate_prompt(request.prompt)
        except ValueError as exc:
            return StartInferenceResponse(False, str(exc))

        with self._service_lock:
            with self._session_lock:
                if self.enabled:
                    return StartInferenceResponse(
                        False,
                        "VLA inference is already running. Call "
                        "/piper_vla_runtime/stop first.",
                    )
                # Invalidate any response that may still be in flight from a
                # previous session before preparing the new recorder/session.
                self.session_generation += 1
                generation = self.session_generation
                self.current_prompt = None
                self._start_observation_versions = None
                self._awaiting_initial_chunk = False
                self._last_hold_reason = None
                self._inference_request.clear()
            self._clear_chunks()

            recording_enabled, recording_reason, controller_dry_run = (
                self._recording_gate()
            )
            if recording_enabled:
                try:
                    # A live run must not begin unless its recorder is ready.
                    self._start_recording(
                        controller_dry_run, prompt, generation
                    )
                except Exception as exc:
                    rospy.logerr(
                        "Recording start failed; control remains disabled: %s",
                        exc,
                    )
                    return StartInferenceResponse(
                        False, "recording start failed: %s" % exc
                    )
            else:
                rospy.loginfo(
                    "Recording disabled for this run: %s "
                    "(runtime_dry_run=%s piper_controller_dry_run=%r)",
                    recording_reason, self.dry, controller_dry_run,
                )

            with self._session_lock:
                self.current_prompt = prompt
                self._start_observation_versions = dict(
                    self._observation_versions
                )
                self._awaiting_initial_chunk = True
                self._initial_inference_pending = True
                self._last_hold_reason = None
                self.enabled = True
                # First inference waits until every observation has received
                # at least one update after this session became active.
                self._inference_request.set()

            self._publish_recorded(
                self.prompt_pub,
                "/vla/current_prompt",
                String(data=prompt),
            )
            message = "Started VLA inference with prompt: %s" % prompt
            rospy.loginfo(
                "VLA session start: generation=%d prompt=%r",
                generation,
                prompt,
            )
            return StartInferenceResponse(True, message)

    def stop(self, _):
        with self._service_lock:
            with self._session_lock:
                stopped_generation = self.session_generation
                stopped_prompt = self.current_prompt
                # Advancing the generation invalidates an inference response
                # even if a new start happens before that response arrives.
                self.session_generation += 1
                self.enabled = False
                self.current_prompt = None
                self._start_observation_versions = None
                self._awaiting_initial_chunk = False
                self._last_hold_reason = None
                self._inference_request.clear()
            self._clear_chunks()
            self._publish_recorded(
                self.hold_pub, "/vla/hold", Bool(data=True)
            )
            self._stop_recording("stop_service")
            rospy.loginfo(
                "VLA session stop: generation=%d prompt=%r; "
                "in-flight responses invalidated by generation=%d",
                stopped_generation,
                stopped_prompt,
                self.session_generation,
            )
            return TriggerResponse(
                True, "control stopped and HOLD requested"
            )

    def ready(self):
        with self._session_lock:
            return all(
                getattr(self, name) is not None
                for name in OBSERVATION_NAMES
            )

    def _inference_session_snapshot(self):
        """Return one immutable request snapshot after fresh observations."""
        with self._session_lock:
            if not self.enabled or self.current_prompt is None:
                return None, "session is not enabled"
            baseline = self._start_observation_versions
            if baseline is None:
                return None, "session observation baseline is unavailable"
            waiting = [
                name for name in OBSERVATION_NAMES
                if self._observation_versions[name] <= baseline[name]
            ]
            if waiting:
                return None, "waiting for post-start observations: %s" % ", ".join(waiting)
            return self.observations.capture(self.session_generation, self.current_prompt, baseline), None

    def image(self, msg, rotate_left_90=False):
        return prepare_image(decode_rgb(msg), self.cfg["observation"]["image_size"],
                             int(rotate_left_90))

    def policy_state(self, state_message=None):
        return adapt_single_state(state_message if state_message is not None else self.state,
                                  self.manifest, self.is_pbl8, self.cfg.get("pbl8", {}))

    def _pbl8_action_error(self, action):
        self._last_pbl8_violation_detail = None
        if not self.is_pbl8:
            return None
        value = np.asarray(action, dtype=np.float64)
        translation = value[:3]
        translation_norm = float(np.linalg.norm(translation))
        if translation_norm > self.pbl8_max_translation_delta_m + 1e-8:
            self._last_pbl8_violation_detail = (
                "action_translation=%s translation_norm=%.9f "
                "max_abs_translation=%.9f limit=%.9f" % (
                    translation.tolist(), translation_norm,
                    float(np.max(np.abs(translation))),
                    self.pbl8_max_translation_delta_m,
                )
            )
            return "pbl8_translation_delta_limit"
        rotation = value[3:6]
        rotation_norm = float(np.linalg.norm(rotation))
        if rotation_norm > self.pbl8_max_rotation_delta_rad + 1e-8:
            self._last_pbl8_violation_detail = (
                "action_rotation=%s rotation_norm=%.9f limit=%.9f" % (
                    rotation.tolist(), rotation_norm,
                    self.pbl8_max_rotation_delta_rad,
                )
            )
            return "pbl8_rotation_delta_limit"
        if not is_within_pbl8_gripper_saturation_margin(value[6]):
            self._last_pbl8_violation_detail = (
                "gripper=%.9f accepted_range=[-1.010000, 1.010000]" %
                float(value[6])
            )
            return "pbl8_gripper_range"
        return None

    def _apply_pbl8_gripper_saturation(self, action):
        """Copy an action and saturate only its normalized PBL8 gripper."""
        applied = list(action)
        if not self.is_pbl8:
            return applied
        raw_gripper = float(applied[6])
        applied = apply_pbl8_gripper_saturation(applied)
        applied_gripper = float(applied[6])
        if applied_gripper != raw_gripper:
            rospy.logwarn_throttle(
                1.0,
                "PBL8 gripper saturated: raw=%.9f applied=%.9f "
                "runtime_range=[%.6f, %.6f]",
                raw_gripper, applied_gripper,
                PBL8_GRIPPER_RUNTIME_MIN, PBL8_GRIPPER_RUNTIME_MAX,
            )
        return applied

    def local_target_z(self, action):
        if self.kind == "ee_absolute": return float(action[2])
        if self.kind == "ee_delta": return float(self.ee.pose.position.z + action[2])
        # Joint actions are FK-gated by the bridge, which reuses PiPER's URDF/Pinocchio model.
        return None

    @staticmethod
    def _mean(values):
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _max(values):
        return max(values) if values else 0.0

    def _clear_chunks(self):
        with self._chunk_lock:
            self.chunks.clear()

    def _stage_chunk(self, actions, request_generation, request=None):
        with self._session_lock, self._chunk_lock:
            return self.chunks.stage(actions, request_generation, request)

    def _next_selected_action(self):
        with self._chunk_lock:
            activating = not self.chunks.active and self.chunks.pending is not None
            selected = self.chunks.select()
            if selected is None:
                return None
            if activating:
                rospy.loginfo("VLA chunk %d activated (%d actions)", self.chunks.chunk_id, self.chunks.length)
                with self._stats_lock:
                    self._chunk_stats["chunk_count"] += 1
            chunk_id, index, action, prefetch = selected
            return chunk_id, index, action.tolist(), prefetch

    def _record_chunk_inference(self, observation_ms, websocket_timing, server_infer_ms, round_trip_ms):
        with self._stats_lock:
            stats = self._chunk_stats
            stats["policy_count"] += 1
            stats["observation_ms"].append(observation_ms)
            stats["ws_send_ms"].append(float(websocket_timing.get("send", 0.0)))
            stats["response_wait_ms"].append(float(websocket_timing.get("response_wait", 0.0)))
            stats["policy_infer_ms"].append(server_infer_ms)
            stats["round_trip_ms"].append(round_trip_ms)
            if observation_ms + round_trip_ms > self.deadline_s * 1000.0:
                stats["policy_deadline_miss"] += 1

    def _request_prefetch(self):
        """Request exactly one next chunk; Event coalesces duplicate requests."""
        with self._session_lock:
            if not self.enabled:
                return
            if not self._inference_request.is_set():
                with self._stats_lock:
                    self._chunk_stats["prefetch_count"] += 1
                self._inference_request.set()

    def _build_policy_observation(self, snapshot):
        """Build one request using the prompt captured with its generation."""
        return snapshot.generation, build_observation(snapshot, self.cfg, self.is_pbl8,
                                                      self.policy_state, self.image)

    def _new_inference_request(self, snapshot):
        with self._session_lock:
            # Consume the allowance on the first request, even if it fails.
            # Each explicit Start resets it; observation waiting does not.
            initial = self._initial_inference_pending
            timeout = (float(self.cfg["runtime"].get("initial_inference_timeout_s", 30.0))
                       if initial else self.client.timeout)
            request = InferenceRequest(snapshot.generation, time.monotonic() + timeout,
                                       self.chunks.session.accepts)
            request.check()
            self._initial_inference_pending = False
            return request

    def _inference_loop(self):
        """Run one policy request per requested execution chunk, never continuously."""
        while not rospy.is_shutdown() and not self._stop_inference.is_set():
            if not self._inference_request.wait(0.1):
                continue
            # Clear before inference: a prefetch request can be queued only by
            # the active chunk's configured prefetch action while this request
            # is in flight. That action occurs once per chunk.
            self._inference_request.clear()
            snapshot, wait_reason = self._inference_session_snapshot()
            if snapshot is None:
                rospy.logwarn_throttle(2.0, "%s", wait_reason)
                # Do not spin. Keep a single request pending until fresh
                # post-start observations exist.
                with self._session_lock:
                    if self.enabled:
                        self._inference_request.set()
                self._stop_inference.wait(0.02)
                continue
            loop_started = time.perf_counter()
            try:
                request = self._new_inference_request(snapshot)
                request_generation, observation = (
                    self._build_policy_observation(snapshot)
                )
                observation_ms = (time.perf_counter() - loop_started) * 1000.0
                request_started = time.perf_counter()
                output = infer_request(self.client, observation, request)
                round_trip_ms = (time.perf_counter() - request_started) * 1000.0
                websocket_timing = self.client.last_timing_ms
                server_infer_ms = float(output.get("server_timing", {}).get("infer_ms", 0.0))
                actions = self.chunks.contract.validate(output["actions"], self.execution_chunk_length,
                                                        self.execution_chunk_length)
                with self._session_lock:
                    request.check()
                    if (
                        not self.enabled
                        or request_generation != self.session_generation
                    ):
                        rospy.loginfo(
                            "Discarding stale inference response: "
                            "request_generation=%d current_generation=%d "
                            "enabled=%s",
                            request_generation,
                            self.session_generation,
                            self.enabled,
                        )
                        continue
                    # Preserve existing /vla/raw_action meaning: policy
                    # response actions[0]. Publish and stage atomically with
                    # respect to start/stop session transitions.
                    self._publish_recorded(
                        self.raw_pub, "/vla/raw_action",
                        Float64MultiArray(data=actions[0].tolist()),
                    )
                    self._publish_recorded(
                        self.type_pub, "/vla/action_type",
                        String(data=self.kind),
                    )
                    accepted = self._stage_chunk(actions, request_generation, request)
                    if accepted:
                        self._record_chunk_inference(
                            observation_ms, websocket_timing,
                            server_infer_ms, round_trip_ms,
                        )
            except (PolicyCommunicationError, StaleResponse) as exc:
                self.client.close()
                rospy.logerr_throttle(2.0, "Inference failed: %s", exc)
            except Exception as exc:
                rospy.logerr_throttle(2.0, "Inference failed: %s", exc)
            # No rate sleep here: the next inference is triggered only by the
            # next active chunk reaching prefetch_start_index.

    def _hold_chunk(self, reason):
        with self._stats_lock:
            if self._last_hold_reason != reason:
                self._chunk_stats["safety_hold_count"] += 1
        self._last_hold_reason = reason
        self._publish_recorded(self.safety_pub, "/safety/state", String(data="HOLD"))
        self._publish_recorded(self.reason_pub, "/safety/reason", String(data=reason))
        self._publish_recorded(self.hold_pub, "/vla/hold", Bool(data=True))
        rospy.logwarn_throttle(1.0, "VLA HOLD: %s", reason)

    def _reject_action_step(self, reason):
        """Report one rejected action without latching HOLD or stopping chunks."""
        self._last_hold_reason = None
        self._publish_recorded(
            self.safety_pub, "/safety/state", String(data="REJECTED")
        )
        self._publish_recorded(
            self.reason_pub, "/safety/reason", String(data=reason)
        )

    def _latch_chunk_underrun(self):
        """Stop the execution state machine; a late response must not resume it."""
        self.enabled = False
        self._awaiting_initial_chunk = False
        self._inference_request.clear()
        self._clear_chunks()
        self._hold_chunk("chunk_underrun")

    def _publish_chunk_action(self, chunk_id, action_index, action):
        self._publish_recorded(
            self.selected_pub, "/vla/selected_action", Float64MultiArray(data=action)
        )
        pbl8_error = self._pbl8_action_error(action)
        if pbl8_error is not None:
            if pbl8_error in (
                "pbl8_translation_delta_limit",
                "pbl8_rotation_delta_limit",
            ):
                rospy.logwarn(
                    "VLA action step rejected: %s\n%s",
                    pbl8_error,
                    self._last_pbl8_violation_detail,
                )
                self._reject_action_step(pbl8_error)
                return False
            rospy.logwarn(
                "VLA HOLD: %s\n%s",
                pbl8_error,
                self._last_pbl8_violation_detail,
            )
            self.enabled = False
            self._awaiting_initial_chunk = False
            self._inference_request.clear()
            self._clear_chunks()
            self._hold_chunk(pbl8_error)
            return False
        applied_action = self._apply_pbl8_gripper_saturation(action)
        # This preserves the existing runtime-only EE Z gate. Joint targets are
        # intentionally left to the unchanged PiPER controller safety path.
        target_z = self.local_target_z(applied_action)
        if target_z is not None:
            self._publish_recorded(self.z_pub, "/safety/target_z", Float64(data=target_z))
            target = copy.deepcopy(self.ee)
            target.header.stamp = rospy.Time.now()
            target.pose.position.z = target_z
            self._publish_recorded(self.target_pub, "/vla/target_ee_pose", target)
            if self.cfg["safety"]["z_enabled"] and target_z < self.zmin:
                self._hold_chunk("target_z_below_min")
                return False
        self._last_hold_reason = None
        self._publish_recorded(
            self.safety_pub, "/safety/state",
            String(data="DRY_RUN" if self.dry else "APPROVED"),
        )
        self._publish_recorded(self.reason_pub, "/safety/reason", String(data=""))
        self._publish_recorded(self.hold_pub, "/vla/hold", Bool(data=False))
        if not self.dry:
            self._publish_recorded(
                self.command_pub, self.cfg["robot"]["approved_action_topic"],
                Float64MultiArray(data=applied_action),
            )
        rospy.logdebug("VLA selected chunk=%d action_index=%d", chunk_id, action_index)
        return True

    def _log_chunk_statistics_if_due(self):
        now = time.perf_counter()
        if now - self._last_statistics_log < self.statistics_period_s:
            return
        with self._chunk_lock:
            active_chunk_id = self._active_chunk_id if self._active_chunk is not None else 0
            next_chunk_id = self._pending_chunk[0] if self._pending_chunk is not None else 0
        with self._stats_lock:
            stats = self._chunk_stats
            elapsed = now - self._stats_started
            policy_rate = stats["policy_count"] / elapsed if elapsed > 0.0 else 0.0
            command_rate = stats["command_count"] / elapsed if elapsed > 0.0 else 0.0
            rospy.loginfo(
                "VLA chunk timing: model_horizon=%d execution_chunk=%d prefetch_start_index=%d "
                "control_target=%.1f Hz active_chunk=%d next_chunk=%d; "
                "policy actual=%.2f Hz inference_count=%d policy_deadline_miss=%d/%d latency mean/max=%.1f/%.1f ms; "
                "control actual=%.2f Hz control_deadline_miss=%d/%d period mean/max=%.1f/%.1f ms; "
                "observation mean/max=%.1f/%.1f ms ws_send mean/max=%.2f/%.2f ms "
                "server_infer mean/max=%.1f/%.1f ms response_wait mean/max=%.1f/%.1f ms; "
                "chunk_count=%d prefetch_count=%d chunk_underrun=%d safety_hold=%d",
                self.model_action_horizon, self.execution_chunk_length, self.prefetch_start_index,
                self.control_rate_hz, active_chunk_id, next_chunk_id,
                policy_rate, stats["policy_count"], stats["policy_deadline_miss"], stats["policy_count"],
                self._mean(stats["round_trip_ms"]), self._max(stats["round_trip_ms"]),
                command_rate, stats["control_deadline_miss"], stats["command_count"],
                self._mean(stats["control_period_ms"]), self._max(stats["control_period_ms"]),
                self._mean(stats["observation_ms"]), self._max(stats["observation_ms"]),
                self._mean(stats["ws_send_ms"]), self._max(stats["ws_send_ms"]),
                self._mean(stats["policy_infer_ms"]), self._max(stats["policy_infer_ms"]),
                self._mean(stats["response_wait_ms"]), self._max(stats["response_wait_ms"]),
                stats["chunk_count"], stats["prefetch_count"], stats["chunk_underrun_count"], stats["safety_hold_count"],
            )
            self._chunk_stats = {
                "policy_count": 0, "policy_deadline_miss": 0, "command_count": 0,
                "control_deadline_miss": 0, "chunk_underrun_count": 0, "safety_hold_count": 0,
                "chunk_count": 0, "prefetch_count": 0,
                "observation_ms": deque(maxlen=500), "ws_send_ms": deque(maxlen=500),
                "response_wait_ms": deque(maxlen=500), "policy_infer_ms": deque(maxlen=500),
                "round_trip_ms": deque(maxlen=500), "control_period_ms": deque(maxlen=500),
            }
            self._stats_started = now
            self._last_statistics_log = now

    def run(self):
        inference_thread = threading.Thread(
            target=self._inference_loop, name="piper_vla_inference", daemon=True
        )
        inference_thread.start()
        rate = rospy.Rate(self.control_rate_hz)
        last_control_start = None
        while not rospy.is_shutdown():
            control_started = time.perf_counter()
            if last_control_start is not None:
                control_period_ms = (control_started - last_control_start) * 1000.0
                with self._stats_lock:
                    self._chunk_stats["control_period_ms"].append(control_period_ms)
                    if control_period_ms > self.control_deadline_s * 1000.0:
                        self._chunk_stats["control_deadline_miss"] += 1
            last_control_start = control_started
            with self._session_lock:
                if self.enabled:
                    selected = self._next_selected_action()
                    if selected is None:
                        if not self._awaiting_initial_chunk:
                            with self._stats_lock:
                                self._chunk_stats["chunk_underrun_count"] += 1
                            self._latch_chunk_underrun()
                    else:
                        chunk_id, action_index, action, prefetch_now = selected
                        if self._publish_chunk_action(chunk_id, action_index, action):
                            with self._stats_lock:
                                self._chunk_stats["command_count"] += 1
                        # A rejected step still consumes its action index.
                        # Preserve the chunk schedule so rejection at the
                        # prefetch index cannot cause a later underrun.
                        if prefetch_now:
                            self._request_prefetch()
            self._log_chunk_statistics_if_due()
            rate.sleep()
        # ROS shutdown / Ctrl-C: stop control state first, then drain and close
        # the recorder so a valid MCAP/NPZ is left whenever ROS permits it.
        with self._session_lock:
            self.session_generation += 1
            self.enabled = False
            self.current_prompt = None
            self._start_observation_versions = None
            self._awaiting_initial_chunk = False
            self._inference_request.clear()
        self._clear_chunks()
        self._stop_inference.set()
        try:
            self._publish_recorded(self.hold_pub, "/vla/hold", Bool(data=True))
        finally:
            self._stop_recording("ros_shutdown")
            self.client.close()


if __name__ == "__main__": Runtime().run()
