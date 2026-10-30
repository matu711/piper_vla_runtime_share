"""ROS-independent dual contracts, observation freshness and execution chunks."""
import time
from piper_vla_common.execution import ActionContract, ChunkQueue
from piper_vla_common.observations import ObservationStore

import numpy as np

JOINT_NAMES = tuple("joint%d" % i for i in range(1, 7)) + ("gripper",)
CAMERAS = ("front", "left_wrist", "right_wrist")
OBSERVATIONS = ("left", "right") + CAMERAS
DELTA_MASK = (True,) * 6 + (False,) + (True,) * 6 + (False,)
POLICY_CONTRACT = {
    "action_type": "joint_absolute", "state_dim": 14, "action_dim": 14,
    "action_horizon": 50, "internal_action_dim": 32,
    "normalization": "quantile", "joint_unit": "rad", "gripper_unit": "m",
    "layout": "left7_right7", "cameras": list(CAMERAS),
}


def vector(values, width):
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (width,) or not np.all(np.isfinite(result)):
        raise ValueError("expected finite vector[%d], got %s" % (width, result.shape))
    return result.copy()


def arm_state(message):
    values = vector(message.position, 7)
    if len(message.name) != 7 or set(message.name) != set(JOINT_NAMES):
        raise ValueError("joint names must be joint1..joint6, gripper")
    return values[[message.name.index(name) for name in JOINT_NAMES]]


def validate_contract(metadata, contract=None):
    for key, expected in (POLICY_CONTRACT if contract is None else contract).items():
        if metadata.get(key) != expected:
            raise ValueError("policy contract mismatch %s: %r != %r" %
                             (key, metadata.get(key), expected))


class Observations(ObservationStore):
    def __init__(self, timeout, skew):
        super().__init__(OBSERVATIONS)
        if not np.isfinite([timeout, skew]).all() or timeout <= 0 or skew < 0:
            raise ValueError("invalid observation timeout/skew")
        self.timeout, self.skew = timeout, skew

    def capture(self, generation, prompt, baseline=None, now=None, ros_now=None):
        return super().capture(generation, prompt, baseline, now, ros_now, self.timeout, self.skew)

    def snapshot(self, now, ros_now, baseline=None):
        snapshot = self.capture(0, "", baseline, now, ros_now)
        state = np.concatenate([arm_state(snapshot.messages[s]) for s in ("left", "right")])
        return state.astype(np.float32), {key: snapshot.messages[key] for key in CAMERAS}


class Chunks(ChunkQueue):
    """Compatibility entry point for the frozen dual action contract."""
    def __init__(self, length=20, prefetch=15):
        super().__init__(length, prefetch, ActionContract(14, 50))
