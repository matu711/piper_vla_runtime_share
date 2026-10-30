#!/usr/bin/env python3
"""Dual-only ROS observation -> WebSocket -> prefetch -> 20 Hz execution."""
import json
import os
import re
import threading
import time
from pathlib import Path

import numpy as np
import rospy
import yaml
from piper_vla_common.policy_client import (InferenceRequest, infer_request,
    PolicyCommunicationError, StaleResponse)
from piper_vla_common.observations import decode_rgb
from piper_vla_common.adapters import prepare_image
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Bool, Float64MultiArray, MultiArrayDimension, String
from std_srvs.srv import Trigger, TriggerResponse
from piper_vla_runtime_msgs.srv import StartInference, StartInferenceResponse

from piper_vla_dual_runtime.core import CAMERAS, Chunks, Observations, validate_contract
from piper_vla_dual_runtime.policy_client import PolicyClient
from piper_vla_dual_runtime.profiles import profile_from_manifest
from piper_vla_dual_runtime.policy_transforms import IMAGE_KEYS, STATE_KEY
from piper_vla_dual_runtime.runtime_recorder import RuntimeRecorder

PREFIX = "/vla_dual"


class Runtime:
    prompt = property(lambda self: self.chunks.session.prompt,
                      lambda self, value: setattr(self.chunks.session, "prompt", value))

    def __init__(self):
        self.cfg = yaml.safe_load(Path(os.environ.get(
            "RUNTIME_CONFIG", "/runtime/config/runtime.dual.yaml")).read_text())
        self.manifest = yaml.safe_load(Path(self.cfg["model"]["manifest"]).read_text())
        self.profile = profile_from_manifest(self.manifest)
        self.policy_contract = self.profile.contract
        validate_contract(self.manifest["contract"], self.policy_contract)
        input_topics = [self.cfg["observation"][name + "_topic"] for name in ("left", "right") + CAMERAS]
        if len(set(input_topics)) != 5:
            raise ValueError("left/right State and three camera topics must be distinct")
        runtime = self.cfg["runtime"]
        runtime.setdefault("controller_status_timeout_s", 0.5)
        if runtime["control_rate_hz"] != 20.0:
            raise ValueError("dual policy requires control_rate_hz=20")
        self.dry = runtime["dry_run"]
        if type(self.dry) is not bool:
            raise ValueError("dry_run must be bool")
        for key in ("inference_timeout_s", "startup_timeout_s", "controller_start_timeout_s",
                    "controller_status_timeout_s"):
            if isinstance(runtime[key], bool) or not np.isfinite(runtime[key]) or runtime[key] <= 0:
                raise ValueError(key + " must be positive")
        self.max_execution_steps = runtime.get("max_execution_steps", 0)
        if type(self.max_execution_steps) is not int or self.max_execution_steps < 0:
            raise ValueError("max_execution_steps must be a nonnegative integer (0=unlimited)")
        self.execution_steps = 0
        self.lock = threading.RLock()
        self.service_lock = threading.Lock()
        self.controller_call_lock = threading.Lock()
        self.controller_token = None
        self.controller_started = None
        self.controller_token_generation = None
        self.controller_token_expires = None
        # Existing Controller tokens have session validity, not a time lease.
        # A declared local TTL can only shorten this validity; never auto-renew.
        self.controller_token_ttl = runtime.get("controller_token_ttl_s")
        if self.controller_token_ttl is not None and (
                isinstance(self.controller_token_ttl, bool)
                or not np.isfinite(self.controller_token_ttl) or self.controller_token_ttl <= 0):
            raise ValueError("controller_token_ttl_s must be positive when declared")
        self.controller_tokens_seen = set()
        self._reset_controller_status()
        self.observations = Observations(self.cfg["observation"]["timeout_s"],
                                         self.cfg["observation"]["max_skew_s"])
        self.chunks = Chunks(runtime["execution_chunk_length"], runtime["prefetch_start_index"])
        self.request = threading.Event()
        self.shutdown_event = threading.Event()
        self.client = PolicyClient(os.environ.get("POLICY_HOST", self.cfg["policy"]["host"]),
                                   int(os.environ.get("POLICY_PORT", self.cfg["policy"]["port"])),
                                   runtime["inference_timeout_s"], contract=self.policy_contract)
        self.recorder = None
        self.fault_reason = None
        self.inflight_generation, self.inflight_since = None, None
        self.inflight_reported = False
        self.last_communication_error = None
        self.prompt = None
        rospy.init_node("piper_vla_dual_runtime")
        self.pubs = {
            name: rospy.Publisher(PREFIX + "/" + name, cls,
                                  queue_size=1 if name in ("approved_action", "hold") else 10,
                                  latch=latch)
            for name, cls, latch in (
                ("raw_action", Float64MultiArray, False),
                ("selected_action", Float64MultiArray, False),
                ("state", Float64MultiArray, False), ("policy_chunk", Float64MultiArray, False),
                ("approved_action", Float64MultiArray, False), ("hold", Bool, True),
                ("safety_state", String, True), ("safety_reason", String, True),
                ("current_prompt", String, True), ("timing", String, False),
                ("execution_step", String, False))}
        self.topics = {}
        self.subscribers = []
        for name in ("left", "right") + CAMERAS:
            topic = self.cfg["observation"][name + "_topic"]
            self.topics[name] = topic
            cls = JointState if name in ("left", "right") else CompressedImage
            self.subscribers.append(rospy.Subscriber(topic, cls, self._callback(name), queue_size=1))
        self.subscribers.append(rospy.Subscriber(PREFIX + "/controller_fault", String,
                                                 self._controller_fault, queue_size=1))
        self.subscribers.append(rospy.Subscriber(PREFIX + "/controller_status", String,
                                                 self._controller_status, queue_size=1))
        for side in ("left", "right"):
            topic = PREFIX + "/" + side + "/target_joint_command"
            self.subscribers.append(rospy.Subscriber(topic, JointState, self._record_callback(topic), queue_size=10))
        rospy.Service("~start", StartInference, self.start)
        rospy.Service("~stop", Trigger, self.stop)
        rospy.on_shutdown(self.shutdown)
        self.pubs["hold"].publish(Bool(data=True))

    def _callback(self, name):
        def receive(message):
            with self.lock:
                self.observations.update(name, message)
                self._record(self.topics[name], message)
        return receive

    def _record_callback(self, topic):
        def receive(message):
            with self.lock:
                self._record(topic, message)
        return receive

    def _controller_fault(self, message):
        with self.lock:
            self._record(PREFIX + "/controller_fault", message)
            if self.chunks.enabled and message.data:
                self._fault("controller:" + message.data)

    def _reset_controller_status(self):
        self.controller_status = None
        self.controller_status_confirmed = False
        self.controller_status_after_ns = None

    def _controller_status(self, message):
        """Accept only fresh, advancing ROS stamps from this start attempt.

        The existing token correlates state with the Trigger ACK. Before the
        first matching RUNNING state, pre-ACK HOLD/old-token packets are normal.
        Ignored/replayed packets never refresh the monotonic watchdog.
        """
        with self.lock:
            try:
                self._record(PREFIX + "/controller_status", message)
                if (self.dry or not self.chunks.enabled
                        or self.controller_status_after_ns is None):
                    return
                data = json.loads(message.data)
                stamp = data["stamp_ns"]
                state, token = data["state"], data["token"]
                if (type(data.get("schema")) is not int or data["schema"] != 1
                        or type(stamp) is not int or stamp <= 0
                        or state not in ("HOLD", "RUNNING", "FAULT")
                        or not isinstance(token, str)
                        or (token and re.fullmatch(r"[0-9a-f]{32}", token) is None)
                        or (state == "RUNNING" and not token)):
                    raise ValueError("controller_status_invalid")
                now_ns = rospy.Time.now().to_nsec()
                timeout_ns = self.cfg["runtime"]["controller_status_timeout_s"] * 1e9
                if (stamp < self.controller_status_after_ns or not 0 <= now_ns - stamp <= timeout_ns
                        or (self.controller_status is not None and stamp <= self.controller_status[0])):
                    return
                now = time.monotonic()
                if self.controller_status_confirmed:
                    # A late heartbeat cannot erase an already elapsed outage
                    # merely by arriving just before the next 20 Hz tick.
                    self._check_controller_status(now)
                self.controller_status = (stamp, now, state, token)
                if self.controller_token is not None:
                    self._check_controller_status(now)
            except Exception as exc:
                if self.chunks.enabled and not self.dry:
                    self._fault("controller_status:" + str(exc))

    def _check_controller_status(self, now):
        """Called under lock before selecting/consuming any live Action."""
        if (not self.controller_status_confirmed and
                now - self.controller_started >= self.cfg["runtime"]["controller_start_timeout_s"]):
            raise ValueError("controller_status_start_timeout")
        timeout = self.cfg["runtime"]["controller_status_timeout_s"]
        status = self.controller_status
        if status is not None:
            stamp, received, state, token = status
            age = (rospy.Time.now().to_nsec() - stamp) / 1e9
            fresh = 0 <= now - received <= timeout and 0 <= age <= timeout
            if fresh:
                if self.controller_status_confirmed or token == self.controller_token:
                    if token != self.controller_token or not token:
                        raise ValueError("controller_status_token_mismatch")
                    if state != "RUNNING":
                        raise ValueError("controller_status_" + state.lower())
                    self.controller_status_confirmed = True
                    return True
        if self.controller_status_confirmed:
            raise ValueError("controller_status_timeout")
        return False

    def _record(self, topic, message):
        if self.recorder is not None:
            header = getattr(message, "header", None)
            stamp = header.stamp.to_nsec() if header is not None else 0
            self.recorder.record(topic, message, stamp or rospy.Time.now().to_nsec())

    def _publish(self, name, message):
        self.pubs[name].publish(message)
        if name == "approved_action":
            # Count the single [left7,right7] ROS publication, not prediction,
            # selection, recorder success, or per-arm Controller/driver commit.
            self.execution_steps += 1
            rospy.loginfo("Dual execution_steps=%d max_execution_steps=%d session=%s",
                          self.execution_steps, self.max_execution_steps, self.session_id)
        self._record(PREFIX + "/" + name, message)

    def _fault(self, reason):
        if self.fault_reason is None:
            self.fault_reason = reason
        self.chunks.stop()
        self.controller_token = None
        self.controller_token_generation = self.controller_token_expires = None
        self.controller_started = None
        self._reset_controller_status()
        self.request.clear()
        self._publish("hold", Bool(data=True))
        self._publish("safety_state", String(data="HOLD"))
        self._publish("safety_reason", String(data=reason))
        if self.recorder is not None:
            self.recorder.metadata.update(execution_steps=self.execution_steps,
                                          max_execution_steps=self.max_execution_steps,
                                          fault_reason=self.fault_reason)
        rospy.logwarn("Dual HOLD: %s; execution_steps=%d max_execution_steps=%d",
                      reason, self.execution_steps, self.max_execution_steps)
        # Keep recorder open to capture subsequent external diagnostics.
        # ~stop or the next ~start finalizes this stopped session.

    def _close_recording(self, reason):
        recorder, self.recorder = self.recorder, None
        if recorder is not None:
            recorder.metadata.update(fault_reason=self.fault_reason,
                                     execution_steps=self.execution_steps,
                                     max_execution_steps=self.max_execution_steps)
            recorder.close(reason)

    def start(self, request):
        prompt = request.prompt
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 2048 or "\x00" in prompt:
            return StartInferenceResponse(False, "prompt must be nonempty, <=2048 chars, without NUL")
        with self.service_lock, self.lock:
            if self.chunks.enabled:
                return StartInferenceResponse(False, "already running; call ~stop first")
            try:
                self._close_recording("next_start")
                self.fault_reason = None
                self.execution_steps = 0
                self.session_id = time.time_ns()
                recording = self.cfg["recording"]
                if recording["enabled"]:
                    self.recorder = RuntimeRecorder(recording["root"], dict(
                        contract=self.policy_contract, config=self.cfg, manifest=self.manifest,
                        prompt=prompt.strip(), session_id=self.session_id,
                        runtime_dry_run=self.dry, execution_steps=0,
                        max_execution_steps=self.max_execution_steps), recording["queue_size"],
                        vector_topics={self.topics[side]: 7 for side in ("left", "right")})
                    self.recorder.start()
                self.prompt = prompt.strip()
                self.baseline = dict(self.observations.versions)
                self.started = time.monotonic()
                self.controller_token = None
                self.controller_token_generation = self.controller_token_expires = None
                self.last_communication_error = None
                self.controller_started = None
                self._reset_controller_status()
                self.chunks.start()
                self.request.set()
                self._publish("current_prompt", String(data=self.prompt))
                self._publish("safety_state", String(data="WAITING"))
                return StartInferenceResponse(True, "dual session started")
            except Exception as exc:
                self._fault("start_failed:" + str(exc))
                self._close_recording("start_failed")
                return StartInferenceResponse(False, str(exc))

    def stop(self, _request):
        with self.service_lock, self.lock:
            self._fault("stop_service")
            self._close_recording("stop_service")
        return TriggerResponse(True, "both arms HOLD; chunks discarded")

    def _snapshot(self):
        return self.observations.snapshot(time.monotonic(), rospy.Time.now().to_sec(), self.baseline)

    def _image(self, message, rotation=0):
        return prepare_image(decode_rgb(message), 224, rotation)

    def _build_observation(self, state, images, prompt):
        observation = {STATE_KEY: state.copy(), "prompt": prompt}
        rotations = self.cfg["observation"]["rotate_ccw_quarter_turns"]
        for name in CAMERAS:
            observation[IMAGE_KEYS[name]] = self._image(images[name], rotations[name])
        return observation

    def _inference_loop(self):
        while not self.shutdown_event.is_set():
            if not self.request.wait(0.05):
                continue
            self.request.clear()
            with self.lock:
                if not self.chunks.enabled:
                    continue
                generation = self.chunks.generation
                try:
                    state, images = self._snapshot()
                    observation_snapshot = self.observations.capture(
                        generation, self.prompt, self.baseline, time.monotonic(), rospy.Time.now().to_sec())
                except ValueError as exc:
                    if self.chunks.awaiting_initial and time.monotonic() - self.started < self.cfg["runtime"]["startup_timeout_s"]:
                        self.request.set()
                        state = None
                    else:
                        self._fault(str(exc))
                        state = None
                if state is not None:
                    prompt = observation_snapshot.prompt
                    self.inflight_generation = generation
                    self.inflight_since = time.monotonic()
                    self.inflight_reported = False
                    inference_request = InferenceRequest(generation, self.inflight_since +
                        self.cfg["runtime"]["inference_timeout_s"], self.chunks.session.accepts)
            if state is None:
                self.shutdown_event.wait(0.02)
                continue
            try:
                observation = self._build_observation(state, images, prompt)
                result = infer_request(self.client, observation, inference_request)
                with self.lock:
                    inference_request.check()
                    if self.chunks.stage(result["actions"], generation, inference_request):
                        self._publish("state", Float64MultiArray(data=state.tolist()))
                        actions = np.asarray(result["actions"])
                        self._publish("raw_action", Float64MultiArray(data=actions[0].tolist()))
                        self._publish("policy_chunk", Float64MultiArray(data=actions.reshape(-1).tolist()))
                        self._publish("timing", String(data=json.dumps(dict(
                            websocket=self.client.last_timing_ms, server=result.get("server_timing", {})))))
            except StaleResponse:
                self.client.close()
            except PolicyCommunicationError as exc:
                with self.lock:
                    if self.chunks.session.accepts(generation):
                        self._communication_error(str(exc))
                self.client.close()
            except Exception as exc:
                with self.lock:
                    if self.chunks.enabled and generation == self.chunks.generation:
                        self._fault("inference:" + str(exc))
                self.client.close()
            finally:
                with self.lock:
                    if self.inflight_generation == generation:
                        self.inflight_since, self.inflight_generation = None, None

    def _controller_start(self, generation):
        """Bound an asynchronous Trigger response to exactly one Runtime start.

        Serialize calls: a late service completion must precede the next start
        at the Controller as well. No ROS callback blocks the control loop.
        """
        try:
            with self.controller_call_lock:
                with self.lock:
                    if not self.chunks.enabled or generation != self.chunks.generation:
                        return
                service = PREFIX + "/controller_start"
                rospy.wait_for_service(service, timeout=self.cfg["runtime"]["controller_start_timeout_s"])
                with self.lock:
                    if not self.chunks.enabled or generation != self.chunks.generation:
                        return
                    # Boundary before the service: a matching status can arrive
                    # before its ACK. Never reuse a sample from an earlier start.
                    self._reset_controller_status()
                    self.controller_status_after_ns = rospy.Time.now().to_nsec()
                response = rospy.ServiceProxy(service, Trigger)()
            with self.lock:
                token = response.message
                valid = (response.success and isinstance(token, str)
                         and re.fullmatch(r"[0-9a-f]{32}", token) is not None
                         and token not in self.controller_tokens_seen)
                if isinstance(token, str):
                    self.controller_tokens_seen.add(token)
                if not self.chunks.enabled or generation != self.chunks.generation:
                    return
                if time.monotonic() - self.controller_started >= self.cfg["runtime"]["controller_start_timeout_s"]:
                    self._fault("controller_start_timeout")
                elif not valid:
                    self._fault("controller_start_invalid_ack")
                else:
                    self.controller_token = token
                    self.controller_token_generation = generation
                    self.controller_token_expires = (None if self.controller_token_ttl is None
                        else time.monotonic() + self.controller_token_ttl)
                    self._check_controller_status(time.monotonic())
        except Exception as exc:
            with self.lock:
                if self.chunks.enabled and generation == self.chunks.generation:
                    self._fault("controller_start:" + str(exc))

    def _communication_error(self, reason):
        """Keep valid chunks/token; no automatic retry or Controller restart."""
        self.last_communication_error = reason
        rospy.logwarn("Dual inference communication error: %s; retaining valid chunks", reason)
        self._publish("timing", String(data=json.dumps(dict(communication_error=reason))))

    def tick(self):
        """One control period, also called directly by Mock tests."""
        with self.lock:
            if not self.chunks.enabled:
                return
            runtime = self.cfg["runtime"]
            now = time.monotonic()
            try:
                if self.recorder and self.recorder.error:
                    raise ValueError("recorder:" + self.recorder.error)
                if (self.inflight_generation == self.chunks.generation and self.inflight_since is not None
                        and now - self.inflight_since > runtime["inference_timeout_s"]):
                    if not self.inflight_reported:
                        self.inflight_reported = True
                        self._communication_error("policy_timeout")
                if not self.dry and self.controller_token is not None:
                    if not self._check_controller_status(now):
                        return
                try:
                    self._snapshot()  # Check freshness every 50 ms, including active chunks.
                except ValueError:
                    if self.chunks.awaiting_initial and now - self.started < runtime["startup_timeout_s"]:
                        return
                    raise
                if self.chunks.awaiting_initial and now - self.started > runtime["startup_timeout_s"]:
                    raise ValueError("initial_chunk_timeout")
                if not self.dry and self.controller_token is not None:
                    if (self.controller_token_generation != self.chunks.generation
                            or re.fullmatch(r"[0-9a-f]{32}", self.controller_token) is None):
                        raise ValueError("controller_token_invalid")
                    if self.controller_token_expires is not None and now >= self.controller_token_expires:
                        raise ValueError("controller_token_expired")
                if not self.dry and self.controller_token is None:
                    if not self.chunks.awaiting_initial:
                        raise ValueError("controller_token_invalid")
                    if self.controller_started is None:
                        # Request release only once the first chunk is ready.
                        if self.chunks.pending is None:
                            return
                        self.controller_started = now
                        threading.Thread(target=self._controller_start,
                                         args=(self.chunks.generation,), daemon=True).start()
                    elif now - self.controller_started >= runtime["controller_start_timeout_s"]:
                        raise ValueError("controller_start_timeout")
                    return
                if self.max_execution_steps and self.execution_steps >= self.max_execution_steps:
                    self._fault("max_execution_steps_reached")
                    return
                selected = self.chunks.select()
                if selected is None:
                    return
                chunk_id, index, action, prefetch = selected
                self._publish("selected_action", Float64MultiArray(data=action.tolist()))
                if not self.dry:
                    # One absolute 14-vector. No per-arm publication, state
                    # addition, CAN or controller logic belongs here.
                    message = Float64MultiArray(data=action.tolist())
                    message.layout.dim = [MultiArrayDimension(
                        label="controller_session:" + self.controller_token,
                        size=14, stride=14)]
                    self._publish("approved_action", message)
                    if not self.chunks.enabled:
                        return  # A synchronous Mock fault must not overwrite HOLD.
                self._publish("execution_step", String(data=json.dumps(dict(
                    session_id=self.session_id, chunk_id=chunk_id, action_index=index,
                    execution_steps=self.execution_steps, max_execution_steps=self.max_execution_steps,
                    sent_to_controller=not self.dry))))
                self._publish("safety_state", String(data="DRY_RUN" if self.dry else "DISPATCHED"))
                if self.max_execution_steps and self.execution_steps >= self.max_execution_steps:
                    self._fault("max_execution_steps_reached")
                    return
                if prefetch:
                    self.request.set()
            except Exception as exc:
                self._fault(str(exc))

    def run(self):
        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.thread.start()
        rate = rospy.Rate(20.0)
        try:
            while not rospy.is_shutdown():
                self.tick()
                rate.sleep()
        finally:
            self.shutdown()

    def shutdown(self):
        self.shutdown_event.set()
        with self.lock:
            self._fault("ros_shutdown")
            self._close_recording("ros_shutdown")


if __name__ == "__main__":
    Runtime().run()
