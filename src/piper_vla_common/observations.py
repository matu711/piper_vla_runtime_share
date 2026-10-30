"""Snapshot bookkeeping shared by single and dual ROS acquisition profiles.

Callers hold their session lock while updating/capturing. ROS messages are
retained by reference: callbacks replace messages and never mutate them.
"""
from dataclasses import dataclass
import time
from types import MappingProxyType

import cv2
import numpy as np


@dataclass(frozen=True)
class ObservationSnapshot:
    generation: int
    prompt: str
    messages: object
    source_stamps: object
    received: object
    versions: object


class ObservationStore:
    def __init__(self, names):
        self.names = tuple(names)
        self.samples = {}
        self.versions = dict.fromkeys(self.names, 0)

    def update(self, name, message, received=None):
        if name not in self.versions:
            raise ValueError("unknown observation " + name)
        self.samples[name] = (message, time.monotonic() if received is None else received)
        self.versions[name] += 1

    def capture(self, generation, prompt, baseline=None, now=None, ros_now=None,
                timeout=None, skew=None):
        if set(self.samples) != set(self.names):
            raise ValueError("observation_missing")
        if baseline is not None and any(self.versions[k] <= baseline[k] for k in self.names):
            raise ValueError("waiting_for_post_start_observations")
        stamps, messages, received = {}, {}, {}
        for name in self.names:
            message, local_time = self.samples[name]
            header = getattr(message, "header", None)
            stamp = float(header.stamp.to_sec()) if header is not None else None
            if timeout is not None:
                if (stamp is None or not np.isfinite(stamp) or stamp <= 0
                        or now - local_time > timeout
                        or not -skew <= ros_now - stamp <= timeout):
                    raise ValueError("observation_stale:" + name)
            messages[name], received[name], stamps[name] = message, local_time, stamp
        if skew is not None and max(stamps.values()) - min(stamps.values()) > skew:
            raise ValueError("observation_time_skew")
        return ObservationSnapshot(generation, prompt, MappingProxyType(messages),
                                   MappingProxyType(stamps), MappingProxyType(received),
                                   MappingProxyType(dict(self.versions)))


def decode_rgb(message):
    image = cv2.imdecode(np.frombuffer(message.data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("invalid_compressed_image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
