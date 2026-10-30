"""Asynchronous ROS1 MCAP + NPZ recorder for ``piper_vla_runtime``."""
from __future__ import annotations

import json
import queue
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from mcap.writer import Writer as McapWriter
from sensor_msgs.msg import JointState


TOPIC_JOINT_STATE = "/joint_states_single"
TOPIC_END_POSE = "/end_pose"
TOPIC_FRONT_CAMERA = "/camera/front/compressed"
TOPIC_WRIST_CAMERA = "/camera/wrist/compressed"
TOPIC_END_POSE_EULER = "/end_pose_euler"
TOPIC_TARGET_JOINT = "/vla/target_joint_command"
TOPIC_TARGET_EE = "/vla/target_ee_command"

NPZ_VECTOR_TOPICS = {
    TOPIC_JOINT_STATE: ("joint_state", 7),
    "/vla/raw_action": ("raw_action", 7),
    "/vla/selected_action": ("selected_action", 7),
    "/vla/approved_action": ("approved_action", 7),
    TOPIC_TARGET_JOINT: ("target_joint", 7),
}
NPZ_STRING_TOPICS = {
    "/safety/state": "safety_state",
    "/safety/reason": "safety_reason",
    "/vla/action_type": "action_type",
}
NPZ_SCALAR_TOPICS = {"/safety/target_z": "safety_target_z"}
CAMERA_TOPICS = {TOPIC_FRONT_CAMERA: "front_camera", TOPIC_WRIST_CAMERA: "wrist_camera"}
REQUIRED_MCAP_TOPICS = (
    TOPIC_JOINT_STATE, TOPIC_END_POSE, TOPIC_END_POSE_EULER,
    TOPIC_FRONT_CAMERA, TOPIC_WRIST_CAMERA,
    "/vla/raw_action", "/vla/selected_action", "/vla/approved_action",
    "/vla/action_type", "/vla/hold", TOPIC_TARGET_JOINT, TOPIC_TARGET_EE,
    "/safety/state", "/safety/reason", "/safety/target_z",
)

# Source definitions from double_PiPER/src/piper_msgs/msg. They allow an MCAP
# reader to decode raw AnyMsg payloads without building the controller here.
RAW_MESSAGE_DEFINITIONS = {
    "piper_msgs/PiperEulerPose": """std_msgs/Header header
float64 x
float64 y
float64 z
float64 roll
float64 pitch
float64 yaw
================================================================================
MSG: std_msgs/Header
uint32 seq
time stamp
string frame_id
""",
    "piper_msgs/PosCmd": """float64 x
float64 y
float64 z
float64 roll
float64 pitch
float64 yaw
float64 gripper
int32 mode1
int32 mode2
""",
}


def message_stamp_ns(message: object) -> int:
    """Use a valid ROS header timestamp, otherwise use the receive ROS time."""
    try:
        stamp_ns = int(message.header.stamp.to_nsec())
        if stamp_ns > 0:
            return stamp_ns
    except Exception:
        pass
    return int(rospy.Time.now().to_nsec())


class _Ros1McapWriter:
    """ROS1 writer compatible with mcap_ros1, including raw ``AnyMsg`` data."""

    def __init__(self, output):
        self._writer = McapWriter(output=output)
        self._writer.start(profile="ros1", library="piper_vla_runtime")
        self._schemas, self._channels = {}, {}
        self._finished = False

    def _channel(self, topic: str, message: object, raw: bool) -> int:
        connection = getattr(message, "_connection_header", {}) or {}
        message_type = connection.get("type") or getattr(message, "_type", "unknown/Unknown")
        channel_key = (topic, message_type, raw)
        if channel_key in self._channels:
            return self._channels[channel_key]
        schema_key = (message_type, raw)
        if schema_key not in self._schemas:
            schema_text = (
                RAW_MESSAGE_DEFINITIONS.get(message_type, "")
                if raw else getattr(message.__class__, "_full_text", "")
            )
            definition = schema_text.encode("utf-8")
            self._schemas[schema_key] = self._writer.register_schema(
                name=message_type, encoding="ros1msg", data=definition
            )
        channel_id = self._writer.register_channel(
            topic=topic, message_encoding="ros1", schema_id=self._schemas[schema_key]
        )
        self._channels[channel_key] = channel_id
        return channel_id

    def write_message(self, topic: str, message: object, stamp_ns: int) -> None:
        raw = isinstance(message, rospy.AnyMsg)
        if raw:
            payload = bytes(message._buff or b"")
        else:
            from io import BytesIO
            buffer = BytesIO()
            message.serialize(buffer)
            payload = buffer.getvalue()
        self._writer.add_message(
            channel_id=self._channel(topic, message, raw),
            log_time=stamp_ns, publish_time=stamp_ns, data=payload,
        )

    def finish(self) -> None:
        if not self._finished:
            self._writer.finish()
            self._finished = True


class RuntimeRecorder:
    """One recorder instance per runtime ``/start``; never reuses a run path."""

    def __init__(self, root: str | Path, metadata: dict, queue_size: int = 512):
        self.root, self.metadata = Path(root), dict(metadata)
        self.queue = queue.Queue(maxsize=int(queue_size))
        self.active, self.run_id, self.run_dir, self.started_at = False, None, None, None
        self._worker, self._mcap_file, self._mcap = None, None, None
        self._subscribers, self._lock = [], threading.Lock()
        self._stop_marker = object()
        self._dropped, self._topic_counts = defaultdict(int), defaultdict(int)
        self._vectors, self._vector_times = defaultdict(list), defaultdict(list)
        self._strings, self._string_times = defaultdict(list), defaultdict(list)
        self._scalars, self._scalar_times = defaultdict(list), defaultdict(list)
        self._camera_times = defaultdict(list)

    @property
    def mcap_path(self): return self.run_dir / (self.run_id + ".mcap")
    @property
    def npz_path(self): return self.run_dir / (self.run_id + ".npz")
    @property
    def metadata_path(self): return self.run_dir / (self.run_id + "_metadata.json")

    def start(self) -> None:
        with self._lock:
            if self.active:
                raise RuntimeError("recorder is already active")
            self.root.mkdir(parents=True, exist_ok=True)
            # Generate exactly once for this run; every path below uses this
            # same start-time identifier.
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = self.root / self.run_id
            self.run_dir.mkdir(mode=0o755)
            self.started_at = datetime.now(timezone.utc)
            self._mcap_file = self.mcap_path.open("wb")
            self._mcap = _Ros1McapWriter(self._mcap_file)
            self.active = True
            self._worker = threading.Thread(
                target=self._writer_loop, name="piper_vla_mcap_writer", daemon=True
            )
            self._worker.start()
            # Custom PiPER messages may not be installed in this image. Save
            # their ROS1 serialized bytes without deserializing them.
            self._subscribers = [
                rospy.Subscriber(TOPIC_END_POSE_EULER, rospy.AnyMsg,
                                 self._subscriber_callback(TOPIC_END_POSE_EULER), queue_size=10),
                rospy.Subscriber(TOPIC_TARGET_JOINT, JointState,
                                 self._subscriber_callback(TOPIC_TARGET_JOINT), queue_size=10),
                rospy.Subscriber(TOPIC_TARGET_EE, rospy.AnyMsg,
                                 self._subscriber_callback(TOPIC_TARGET_EE), queue_size=10),
            ]

    def _subscriber_callback(self, topic: str):
        return lambda message: self.record(topic, message)

    def record(self, topic: str, message: object) -> bool:
        """Non-blocking producer API for ROS callbacks and the control loop."""
        if not self.active:
            return False
        try:
            self.queue.put_nowait((topic, message, message_stamp_ns(message)))
            return True
        except queue.Full:
            self._dropped[topic] += 1
            rospy.logwarn_throttle(2.0, "Recorder queue full; dropping %s", topic)
            return False

    @staticmethod
    def _vector_from_message(message: object):
        values = getattr(message, "position", None)
        if values is None:
            values = getattr(message, "data", None)
        if values is None or isinstance(values, (str, bytes)):
            return None
        try:
            return np.asarray(values, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _pose_from_message(message: PoseStamped):
        pose = message.pose
        return np.asarray([
            pose.position.x, pose.position.y, pose.position.z,
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w,
        ], dtype=np.float64)

    def _append_npz(self, topic: str, message: object, stamp_ns: int) -> None:
        stamp_s = stamp_ns / 1_000_000_000.0
        if topic in NPZ_VECTOR_TOPICS:
            key, expected_size = NPZ_VECTOR_TOPICS[topic]
            vector = self._vector_from_message(message)
            if vector is None or vector.size != expected_size:
                rospy.logwarn_throttle(2.0, "Recorder ignored malformed %s vector", topic)
                return
            self._vector_times[key].append(stamp_s)
            self._vectors[key].append(vector)
        elif topic == TOPIC_END_POSE:
            self._vector_times["end_pose"].append(stamp_s)
            self._vectors["end_pose"].append(self._pose_from_message(message))
        elif topic in NPZ_STRING_TOPICS:
            key = NPZ_STRING_TOPICS[topic]
            self._string_times[key].append(stamp_s)
            self._strings[key].append(str(getattr(message, "data", "")))
        elif topic in NPZ_SCALAR_TOPICS:
            key = NPZ_SCALAR_TOPICS[topic]
            self._scalar_times[key].append(stamp_s)
            self._scalars[key].append(float(getattr(message, "data", np.nan)))
        elif topic in CAMERA_TOPICS:
            self._camera_times[CAMERA_TOPICS[topic]].append(stamp_s)

    def _writer_loop(self) -> None:
        while True:
            item = self.queue.get()
            try:
                if item is self._stop_marker:
                    return
                topic, message, stamp_ns = item
                self._mcap.write_message(topic, message, stamp_ns)
                self._topic_counts[topic] += 1
                self._append_npz(topic, message, stamp_ns)
            except Exception as exc:
                rospy.logerr("Recorder write failed: %s", exc)
            finally:
                self.queue.task_done()

    def _npz_payload(self) -> dict:
        payload = {}
        for key, width in (("joint_state", 7), ("end_pose", 7), ("raw_action", 7),
                           ("selected_action", 7), ("approved_action", 7), ("target_joint", 7)):
            payload[key + "_t"] = np.asarray(self._vector_times[key], dtype=np.float64)
            payload[key] = (np.asarray(self._vectors[key], dtype=np.float64).reshape(-1, width)
                            if self._vectors[key] else np.empty((0, width), dtype=np.float64))
        for key in ("safety_state", "safety_reason", "action_type"):
            payload[key + "_t"] = np.asarray(self._string_times[key], dtype=np.float64)
            payload[key] = np.asarray(self._strings[key], dtype=np.str_)
        payload["safety_target_z_t"] = np.asarray(self._scalar_times["safety_target_z"], dtype=np.float64)
        payload["safety_target_z"] = np.asarray(self._scalars["safety_target_z"], dtype=np.float64)
        for key in ("front_camera", "wrist_camera"):
            payload[key + "_t"] = np.asarray(self._camera_times[key], dtype=np.float64)
        return payload

    def close(self, reason: str = "stop") -> dict:
        with self._lock:
            if not self.active:
                return {"run_dir": None, "samples": 0}
            self.active = False
            subscribers, self._subscribers = self._subscribers, []
        for subscriber in subscribers:
            subscriber.unregister()
        self.queue.put(self._stop_marker)
        self._worker.join()
        self._mcap.finish()
        self._mcap_file.close()
        np.savez_compressed(self.npz_path, **self._npz_payload())
        ended_at = datetime.now(timezone.utc)
        metadata = dict(self.metadata)
        metadata.update({
            "start_time": self.started_at.isoformat(),
            "end_time": ended_at.isoformat(),
            "duration_s": (ended_at - self.started_at).total_seconds(),
            "stop_reason": reason,
            "mcap_topics": list(REQUIRED_MCAP_TOPICS),
            "topic_message_counts": dict(self._topic_counts),
            "dropped_messages": dict(self._dropped),
        })
        self.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        return {"run_dir": str(self.run_dir), "mcap": str(self.mcap_path), "npz": str(self.npz_path),
                "samples": int(sum(self._topic_counts.values())), "duration_s": metadata["duration_s"]}
