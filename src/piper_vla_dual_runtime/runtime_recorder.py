"""Per-session asynchronous MCAP + NPZ + metadata, dual-only topic schema."""
import io
import json
import queue
import threading
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from mcap.writer import Writer

PREFIX = "/vla_dual"
VECTOR_WIDTHS = {
    PREFIX + "/state": 14, PREFIX + "/raw_action": 14,
    PREFIX + "/selected_action": 14, PREFIX + "/approved_action": 14,
    PREFIX + "/policy_chunk": 700,
    "/followerL/joint_states_single": 7, "/followerR/joint_states_single": 7,
    PREFIX + "/left/target_joint_command": 7,
    PREFIX + "/right/target_joint_command": 7,
}


class RuntimeRecorder:
    def __init__(self, root, metadata, queue_size=512, vector_topics=None):
        if not isinstance(queue_size, int) or queue_size < 1:
            raise ValueError("recording queue_size must be a positive integer")
        self.root, self.metadata = Path(root), dict(metadata)
        self.vector_widths = dict(VECTOR_WIDTHS)
        self.vector_widths.update(vector_topics or {})
        self.queue = queue.Queue(maxsize=queue_size)
        self.active, self.error = False, None
        self.dropped, self.counts = Counter(), Counter()
        self.vectors, self.strings, self.times = defaultdict(list), defaultdict(list), defaultdict(list)
        self.lock = threading.Lock()
        self.marker = object()

    def start(self):
        self.started = datetime.now(timezone.utc)
        run_id = self.started.strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid.uuid4().hex[:8]
        self.run_dir = self.root / run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.file = (self.run_dir / (run_id + ".mcap")).open("wb")
        self.writer = Writer(self.file)
        self.writer.start(profile="ros1", library="piper_vla_dual_runtime")
        self.channels = {}
        self.active = True
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()

    def record(self, topic, message, stamp_ns=None):
        # Serialize in producer to freeze mutable ROS objects before enqueue.
        if not self.active or self.error:
            return False
        buffer = io.BytesIO()
        message.serialize(buffer)
        values = getattr(message, "position", getattr(message, "data", None))
        vector = None
        if topic in self.vector_widths:
            vector = np.asarray(values, dtype=np.float64).reshape(-1).copy()
            if vector.size != self.vector_widths[topic]:
                self.error = "invalid recorder vector " + topic
                return False
        header = getattr(message, "header", None)
        stamp = header.stamp.to_nsec() if header is not None else 0
        stamp = stamp_ns if stamp_ns is not None else stamp or time.time_ns()
        item = (topic, message._type, message._full_text, buffer.getvalue(), stamp,
                vector, values if isinstance(values, str) else None)
        with self.lock:
            if not self.active:
                return False
            try:
                self.queue.put_nowait(item)
                return True
            except queue.Full:
                self.dropped[topic] += 1
                # Image drops are observable; loss of state/action/safety stops a run.
                if not topic.endswith("/compressed"):
                    self.error = "recorder_queue_full:" + topic
                return False

    def _loop(self):
        while True:
            item = self.queue.get()
            try:
                if item is self.marker:
                    return
                topic, msg_type, definition, payload, stamp, values, string = item
                if topic not in self.channels:
                    schema = self.writer.register_schema(msg_type, "ros1msg", definition.encode())
                    self.channels[topic] = self.writer.register_channel(topic, "ros1", schema)
                self.writer.add_message(self.channels[topic], stamp, payload, publish_time=stamp)
                self.counts[topic] += 1
                key = topic.strip("/").replace("/", "__")
                self.times[key].append(stamp / 1e9)
                if values is not None:
                    self.vectors[key].append(values)
                if string is not None:
                    self.strings[key].append(string)
            except Exception as exc:
                self.error = str(exc)
            finally:
                self.queue.task_done()

    def close(self, reason="stop"):
        with self.lock:
            if not self.active:
                return
            self.active = False
        self.queue.put(self.marker)
        self.worker.join()
        self.writer.finish()
        self.file.close()
        payload = {key + "_t": np.asarray(value) for key, value in self.times.items()}
        for topic, width in self.vector_widths.items():
            key = topic.strip("/").replace("/", "__")
            values = np.asarray(self.vectors[key], dtype=np.float64).reshape(-1, width)
            payload[key] = values.reshape(-1, 50, 14) if width == 700 else values
        payload.update({key: np.asarray(value, dtype=np.str_) for key, value in self.strings.items()})
        np.savez_compressed(self.run_dir / (self.run_dir.name + ".npz"), **payload)
        metadata = dict(self.metadata, start_time=self.started.isoformat(),
                        end_time=datetime.now(timezone.utc).isoformat(), stop_reason=reason,
                        topic_counts=dict(self.counts), dropped_messages=dict(self.dropped),
                        recorder_error=self.error)
        (self.run_dir / (self.run_dir.name + "_metadata.json")).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n")
