#!/usr/bin/env python3
"""ROS1 integration harness for an isolated, device-free mock launch only.

Uses the real Runtime process/services and GPU Policy. Source trees are mounted
read-only; synthetic camera messages and a transparent fault proxy live here.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import rospy
import rosnode
import yaml
from sensor_msgs.msg import JointState, CompressedImage
from std_msgs.msg import Bool, Float64MultiArray, MultiArrayDimension, String
from std_srvs.srv import Trigger
from piper_vla_runtime_msgs.srv import StartInference
from websockets.sync.client import connect
from websockets.sync.server import serve
from openpi_client import msgpack_numpy

RESULTS = os.environ.get("DUAL_RESULTS_DIR", "/results")


class Proxy:
    def __init__(self):
        self.lock = threading.Lock()
        self.mode, self.phase, self.calls = "normal", "setup", 0
        self.records, self.connections = [], []
        self.server = serve(self.handle, "127.0.0.1", 8002, compression=None, max_size=None)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def configure(self, mode, phase):
        with self.lock:
            self.mode, self.phase, self.calls = mode, phase, 0

    def handle(self, client):
        try:
            with connect("ws://127.0.0.1:8001", compression=None, max_size=None, close_timeout=.2) as backend:
                with self.lock:
                    self.connections.append(client)
                client.send(backend.recv(timeout=5))
                for payload in client:
                    started = time.monotonic()
                    with self.lock:
                        mode, phase, index = self.mode, self.phase, self.calls
                        self.calls += 1
                    obs = msgpack_numpy.unpackb(payload)
                    delay = .65 if mode == "underrun" and index >= 1 else (2.6 if mode == "timeout" else 0)
                    record = dict(phase=phase, index=index, started=started, artificial_delay_s=delay,
                                  state_shape=list(obs["observation/state"].shape), prompt=obs["prompt"],
                                  state=obs["observation/state"].tolist(), state_dtype=str(obs["observation/state"].dtype),
                                  image_shapes={key: list(value.shape) for key, value in obs.items() if "image" in key})
                    with self.lock:
                        self.records.append(record)
                    if mode == "disconnect":
                        client.close()
                        return
                    if delay:
                        time.sleep(delay)
                    backend.send(payload)  # Forward wire payload unchanged.
                    response = backend.recv(timeout=10)
                    output = msgpack_numpy.unpackb(response)
                    record.update(finished=time.monotonic(), shape=list(output["actions"].shape),
                                  finite=bool(np.isfinite(output["actions"]).all()), server_timing=output.get("server_timing"),
                                  execution_actions=output["actions"][:20].tolist())
                    client.send(response)
        except Exception as exc:
            # Expected during timeout/connection fault injection or process stop.
            with self.lock:
                self.records.append(dict(phase=self.phase, transport_closed=type(exc).__name__))
        finally:
            client.close()

    def close(self):
        for connection in self.connections:
            connection.close()
        self.server.shutdown()
        self.thread.join(2)


class Harness:
    def __init__(self):
        if os.environ.get("ROS_MASTER_URI") != "http://127.0.0.1:11311":
            raise RuntimeError("isolated loopback ROS master required")
        devices = [path.name for path in Path("/sys/class/net").iterdir()]
        if devices != ["lo"] or list(Path("/dev").glob("video*")) or list(Path("/dev").glob("can*")):
            raise RuntimeError("mock test requires only loopback and no camera/CAN devices")
        assert rospy.get_param("/vla_dual_piper_controller/dry_run") is True
        self.external = os.environ.get("DUAL_EXTERNAL_RUNTIME") == "1"
        expected = {"/followerL/mock_driver", "/followerR/mock_driver", "/rosout", "/vla_dual_piper_controller"}
        if self.external:
            expected.add("/piper_vla_dual_runtime")
        assert set(rosnode.get_node_names()) == expected
        rospy.init_node("dual_mock_integration", anonymous=False)
        self.lock = threading.Lock()
        self.events, self.subscribers, self.runtime = [], [], None
        self.pause_camera, self.done = None, threading.Event()
        self.report = dict(software_only=True, controller_dry_run=True, mock_driver=True,
                           runtime_send_authorized=True, initial_nodes=sorted(expected), phases=[])
        self.proxy = Proxy()
        self.ack_delay = 0
        self.acks = []
        self.ack_service = rospy.Service("/integration/controller_start", Trigger, self.controller_start)
        topics = {
            "/vla/dual_joint_states": JointState,
            "/followerL/joint_states_single": JointState, "/followerR/joint_states_single": JointState,
            "/followerL/joint_states": JointState, "/followerR/joint_states": JointState,
            "/vla_dual/raw_action": Float64MultiArray, "/vla_dual/selected_action": Float64MultiArray,
            "/vla_dual/approved_action": Float64MultiArray, "/vla_dual/state": Float64MultiArray,
            "/vla_dual/hold": Bool, "/vla_dual/execution_step": String, "/vla_dual/timing": String,
            "/vla_dual/safety_state": String, "/vla_dual/safety_reason": String,
            "/vla_dual/current_prompt": String, "/vla_dual/controller_fault": String,
            "/safety/state": String, "/safety/reason": String,
        }
        for side in ("left", "right"):
            topics["/vla_dual/" + side + "/target_joint_command"] = JointState
            topics["/safety/" + side + "/state"] = String
        for camera in ("front", "left_wrist", "right_wrist"):
            topics["/camera/" + camera + "/compressed"] = CompressedImage
        for topic, cls in topics.items():
            self.subscribers.append(rospy.Subscriber(topic, cls, self.receive, callback_args=topic, queue_size=2000))
        self.camera_pubs, self.jpegs = {}, {}
        for index, camera in enumerate(("front", "left_wrist", "right_wrist")):
            self.camera_pubs[camera] = rospy.Publisher("/camera/" + camera + "/compressed", CompressedImage, queue_size=1)
            image = np.full((224, 224, 3), [30 + index * 20, 60, 100], np.uint8)
            ok, encoded = cv2.imencode(".jpg", image)
            assert ok
            self.jpegs[camera] = encoded.tobytes()
        self.camera_thread = threading.Thread(target=self.publish_cameras, daemon=True)
        self.camera_thread.start()
        self.start_service = rospy.ServiceProxy("/piper_vla_dual_runtime/start", StartInference)
        self.stop_service = rospy.ServiceProxy("/piper_vla_dual_runtime/stop", Trigger)

    def controller_start(self, request):
        started, delay = time.monotonic(), self.ack_delay
        response = rospy.ServiceProxy("/vla_dual/controller_start", Trigger)(request)
        if delay:
            time.sleep(delay)
        with self.lock:
            self.acks.append(dict(started=started, returned=time.monotonic(),
                                  success=response.success, token=response.message, delay_s=delay))
        return response

    def receive(self, message, topic):
        values = getattr(message, "data", None)
        if isinstance(message, CompressedImage):
            values = dict(bytes=len(message.data), stamp=message.header.stamp.to_sec())
        elif isinstance(message, JointState):
            values = list(message.position)
        elif isinstance(message, Float64MultiArray):
            values = [float(value) if np.isfinite(value) else None for value in message.data]
        entry = dict(t=time.monotonic(), topic=topic, value=values,
                     caller=message._connection_header.get("callerid"))
        if isinstance(message, Float64MultiArray):
            entry["token_labels"] = [dim.label for dim in message.layout.dim]
        with self.lock:
            self.events.append(entry)

    def publish_cameras(self):
        while not self.done.is_set():
            stamp = rospy.Time.now()
            for camera, pub in self.camera_pubs.items():
                if camera != self.pause_camera:
                    message = CompressedImage()
                    message.header.stamp, message.format, message.data = stamp, "jpeg", self.jpegs[camera]
                    pub.publish(message)
            self.done.wait(.05)

    def subset(self, topic, since, until=float("inf")):
        with self.lock:
            return [event for event in self.events if event["topic"] == topic and since <= event["t"] <= until]

    def wait(self, predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            if self.runtime is not None and self.runtime.poll() is not None:
                raise RuntimeError("Runtime exited; inspect runtime log")
            time.sleep(.02)
        raise TimeoutError("integration condition timed out")

    def fixture_state(self):
        # Initialize *only* the mock drivers using their documented ROS input.
        stats = json.loads(Path("/bundle/checkpoint/assets/local/dnp1_20261015/norm_stats.json").read_text())
        state = stats["norm_stats"]["state"]["mean"]
        pubs = [rospy.Publisher("/follower" + side + "/joint_states", JointState, queue_size=1) for side in ("L", "R")]
        self.wait(lambda: all(pub.get_num_connections() >= 1 for pub in pubs))
        for index, pub in enumerate(pubs):
            message = JointState()
            message.header.stamp = rospy.Time.now()
            message.name = ["joint%d" % i for i in range(1, 7)] + ["gripper"]
            message.position = state[index * 7:(index + 1) * 7]
            pub.publish(message)
        time.sleep(.25)
        self.report["mock_state_initialization"] = dict(source="checkpoint State mean", position14=state,
                                                        fixture_messages=2, physical_driver_present=False)

    def launch_runtime(self, dry):
        if self.external:
            rospy.wait_for_service("/piper_vla_dual_runtime/start", timeout=10)
            rospy.wait_for_service("/piper_vla_dual_runtime/stop", timeout=10)
            return
        config = yaml.safe_load(Path("/runtime/configs/runtime.dual.yaml").read_text())
        config["model"].update(checkpoint="/bundle", manifest="/bundle/manifest.yaml")
        config["policy"]["port"] = 8002
        config["runtime"]["dry_run"] = dry
        config["recording"]["root"] = RESULTS + "/recordings"
        path = Path("/tmp/runtime_dual_integration.yaml")
        path.write_text(yaml.safe_dump(config))
        env = dict(os.environ, RUNTIME_CONFIG=str(path), POLICY_HOST="127.0.0.1", POLICY_PORT="8002")
        log = open(RESULTS + "/runtime_%s.log" % ("dry" if dry else "send"), "w")
        self.runtime = subprocess.Popen([sys.executable, "/runtime/src/piper_vla_dual_runtime/node.py",
                                         "/vla_dual/controller_start:=/integration/controller_start"], env=env, stdout=log, stderr=log)
        self.wait(lambda: "/piper_vla_dual_runtime" in rosnode.get_node_names(), 10)
        rospy.wait_for_service("/piper_vla_dual_runtime/start", timeout=10)
        rospy.wait_for_service("/piper_vla_dual_runtime/stop", timeout=10)
        time.sleep(.3)

    def close_runtime(self):
        if self.runtime is not None:
            self.runtime.send_signal(signal.SIGINT)
            self.runtime.wait(timeout=10)
            self.runtime = None
            time.sleep(.2)

    def start(self, phase, mode="normal"):
        self.proxy.configure(mode, phase)
        started = time.monotonic()
        result = self.start_service("Assembling a box.")
        if not result.success:
            raise RuntimeError(result.message)
        return started

    def stop(self):
        result = self.stop_service()
        if not result.success:
            raise RuntimeError(result.message)
        time.sleep(.25)

    def summary(self, name, since):
        selected = self.subset("/vla_dual/selected_action", since)
        approved = self.subset("/vla_dual/approved_action", since)
        steps = [json.loads(event["value"]) for event in self.subset("/vla_dual/execution_step", since)]
        periods = np.diff([event["t"] for event in selected]) * 1000
        gaps = []
        for previous, current in zip(steps, steps[1:]):
            expected = ((previous["chunk_id"], previous["action_index"] + 1)
                        if previous["action_index"] < 19 else (previous["chunk_id"] + 1, 0))
            if (current["chunk_id"], current["action_index"]) != expected:
                gaps.append([previous, current])
        entry = dict(name=name, selected_count=len(selected), approved_count=len(approved),
                     execution_count=len(steps), index_gaps=gaps,
                     chunks=len(set(step["chunk_id"] for step in steps)),
                     action_shapes_valid=all(len(event["value"]) == 14 for event in selected + approved),
                     runtime_reasons=[event["value"] for event in self.subset("/vla_dual/safety_reason", since)],
                     left_target_count=len(self.subset("/vla_dual/left/target_joint_command", since)),
                     right_target_count=len(self.subset("/vla_dual/right/target_joint_command", since)),
                     left_commit_count=sum(event["value"] == "DRY_RUN" for event in self.subset("/safety/left/state", since)),
                     right_commit_count=sum(event["value"] == "DRY_RUN" for event in self.subset("/safety/right/state", since)))
        if len(periods):
            entry["control_period_ms"] = dict(mean=float(periods.mean()), p95=float(np.percentile(periods, 95)), max=float(periods.max()))
            entry["measured_hz"] = float(1000 / periods.mean())
        entry["approved_equals_selected"] = (len(approved) == len(selected) and all(
            left["value"] == right["value"] for left, right in zip(selected, approved)))
        acknowledgements = [ack for ack in self.acks if since <= ack["started"]]
        runtime_actions = [event for event in approved if event["caller"] == "/piper_vla_dual_runtime"]
        entry["acknowledgements"] = acknowledgements
        entry["runtime_approved_count"] = len(runtime_actions)
        entry["tokens_match_ack"] = all(any(
            ack["success"] and ack["returned"] <= event["t"]
            and event["token_labels"] == ["controller_session:" + ack["token"]]
            for ack in acknowledgements) for event in runtime_actions)
        self.report["phases"].append(entry)
        print(json.dumps(entry), flush=True)
        return entry

    def await_fault(self, since, text=None):
        self.wait(lambda: any((text is None or text in event["value"])
                             for event in self.subset("/vla_dual/safety_reason", since)), 6)
        count = len(self.subset("/vla_dual/selected_action", since))
        time.sleep(.7)
        assert count == len(self.subset("/vla_dual/selected_action", since)), "actions resumed without start"
        assert self.subset("/vla_dual/hold", since)[-1]["value"] is True

    def run(self):
        self.fixture_state()
        if not self.external or os.environ.get("DUAL_EXTERNAL_DRY") == "1":
            self.launch_runtime(True)
            since = self.start("runtime_dry")
            self.wait(lambda: len(self.subset("/vla_dual/selected_action", since)) >= 40)
            self.stop()
            dry = self.summary("runtime_dry", since)
            assert dry["approved_count"] == 0 and not dry["acknowledgements"]
            assert all(event["value"] is True for event in self.subset("/vla_dual/hold", since))
            self.close_runtime()
            if self.external:
                self.report["completed"] = True
                return

        self.launch_runtime(False)
        since = self.start("normal_20hz")
        time.sleep(10)
        self.stop()
        normal = self.summary("normal_20hz", since)
        assert normal["selected_count"] >= 150 and not normal["index_gaps"]
        assert normal["approved_equals_selected"]
        assert 19 <= normal["measured_hz"] <= 21
        assert normal["runtime_approved_count"] == normal["left_commit_count"] == normal["right_commit_count"]
        assert normal["tokens_match_ack"]
        outputs = {record["index"] + 1: record["execution_actions"] for record in self.proxy.records
                   if record.get("phase") == "normal_20hz" and "execution_actions" in record}
        approved = self.subset("/vla_dual/approved_action", since)
        steps = self.subset("/vla_dual/execution_step", since)
        for event, step_event in zip(approved, steps):
            step = json.loads(step_event["value"])
            assert event["value"] == outputs[step["chunk_id"]][step["action_index"]]
        normal["absolute_actions_match_policy"] = True
        assert self.subset("/vla_dual/left/target_joint_command", since)[0]["value"] == self.subset("/vla_dual/approved_action", since)[0]["value"][:7]
        assert self.subset("/vla_dual/right/target_joint_command", since)[0]["value"] == self.subset("/vla_dual/approved_action", since)[0]["value"][7:]

        self.ack_delay = .3
        since = self.start("controller_ack_delay")
        self.wait(lambda: len(self.subset("/vla_dual/approved_action", since)) >= 20)
        self.stop()
        delayed = self.summary("controller_ack_delay", since)
        assert delayed["tokens_match_ack"] and delayed["approved_count"] == delayed["left_commit_count"] == delayed["right_commit_count"]
        self.ack_delay = 2.6
        since = self.start("controller_ack_timeout")
        self.await_fault(since, "controller_start_timeout")
        time.sleep(.5)
        self.stop()
        expired = self.summary("controller_ack_timeout", since)
        assert expired["approved_count"] == expired["selected_count"] == 0
        self.ack_delay = 0

        since = self.start("controller_stop")
        self.wait(lambda: len(self.subset("/vla_dual/approved_action", since)) >= 10)
        rospy.ServiceProxy("/vla_dual_piper_controller/stop", Trigger)()
        self.await_fault(since, "controller:controller_stop")
        self.stop()
        self.summary("controller_stop", since)

        since = self.start("controller_validation_rejection")
        self.wait(lambda: len(self.subset("/vla_dual/approved_action", since)) >= 10)
        invalid_pub = rospy.Publisher("/vla_dual/approved_action", Float64MultiArray, queue_size=1)
        self.wait(lambda: invalid_pub.get_num_connections() >= 1)
        injected = time.monotonic()
        bad = list(self.report["mock_state_initialization"]["position14"])
        bad[7] = float("nan")
        message = Float64MultiArray(data=bad)
        token = self.subset("/vla_dual/approved_action", since)[-1]["token_labels"][0]
        message.layout.dim = [MultiArrayDimension(label=token, size=14, stride=14)]
        invalid_pub.publish(message)
        self.wait(lambda: any("finite" in event["value"] for event in self.subset("/safety/reason", injected)))
        self.await_fault(injected, "controller:")
        self.report["controller_validation_behavior"] = dict(
            injected="one external malformed Action (right joint1 NaN), current session token",
            runtime_continued=False,
            runtime_fault_notifications=len(self.subset("/vla_dual/controller_fault", injected)),
            controller_reasons=[event["value"] for event in self.subset("/safety/reason", injected) if event["value"]])
        assert self.report["controller_validation_behavior"]["runtime_fault_notifications"] >= 1
        invalid_pub.unregister()
        self.stop()
        self.summary("controller_validation_rejection", since)

        since = self.start("camera_missing")
        self.wait(lambda: len(self.subset("/vla_dual/approved_action", since)) >= 10)
        self.pause_camera = "right_wrist"
        self.await_fault(since, "observation_")
        self.pause_camera = None
        time.sleep(.7)
        count = len(self.subset("/vla_dual/selected_action", since))
        time.sleep(.2)
        assert count == len(self.subset("/vla_dual/selected_action", since))
        self.stop()
        self.summary("camera_missing", since)

        since = self.start("chunk_underrun", "underrun")
        self.await_fault(since, "chunk_underrun")
        self.stop()
        self.summary("chunk_underrun", since)

        since = self.start("policy_timeout", "timeout")
        self.await_fault(since)
        self.stop()
        self.summary("policy_timeout", since)
        time.sleep(.5)

        since = self.start("policy_disconnect", "disconnect")
        self.await_fault(since)
        self.stop()
        self.summary("policy_disconnect", since)

        since = self.start("restart_after_faults")
        self.wait(lambda: len(self.subset("/vla_dual/approved_action", since)) >= 30)
        self.stop()
        restarted = self.summary("restart_after_faults", since)
        assert restarted["tokens_match_ack"]
        assert restarted["approved_count"] == restarted["left_commit_count"] == restarted["right_commit_count"]
        tokens = [ack["token"] for ack in self.acks]
        assert len(tokens) == len(set(tokens))
        assert not [event for event in self.events if event["topic"] in (
            "/followerL/joint_states", "/followerR/joint_states") and event["caller"] == "/vla_dual_piper_controller"]
        self.report["completed"] = True

    def close(self):
        try:
            if self.external or (self.runtime is not None and self.runtime.poll() is None):
                self.stop()
                self.close_runtime()
        finally:
            self.done.set()
            self.camera_thread.join(2)
            self.proxy.close()
            self.report["controller_start_acks"] = self.acks
            self.report["policy_requests"] = self.proxy.records
            self.report["controller_driver_commands"] = [event for event in self.events if event["topic"] in (
                "/followerL/joint_states", "/followerR/joint_states") and event["caller"] == "/vla_dual_piper_controller"]
            self.report["observations"] = {topic: len([event for event in self.events if event["topic"] == topic])
                                           for topic in ("/followerL/joint_states_single", "/followerR/joint_states_single",
                                                         "/vla/dual_joint_states", "/camera/front/compressed",
                                                         "/camera/left_wrist/compressed", "/camera/right_wrist/compressed")}
            Path(RESULTS + "/integration.json").write_text(json.dumps(self.report, indent=2) + "\n")
            Path(RESULTS + "/events.json").write_text(json.dumps(self.events) + "\n")


if __name__ == "__main__":
    harness = Harness()
    try:
        harness.run()
    except Exception as exc:
        harness.report["error"] = "%s: %s" % (type(exc).__name__, exc)
        raise
    finally:
        harness.close()
