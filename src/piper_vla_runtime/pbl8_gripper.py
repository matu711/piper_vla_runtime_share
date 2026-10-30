"""PBL8 normalized gripper validation and application helpers."""
import math


PBL8_GRIPPER_RUNTIME_MIN = -1.0
PBL8_GRIPPER_RUNTIME_MAX = 1.0
PBL8_GRIPPER_SATURATION_MARGIN = 0.01


def is_within_pbl8_gripper_saturation_margin(value):
    """Return whether a finite policy value may be safely saturated."""
    value = float(value)
    return (
        math.isfinite(value)
        and PBL8_GRIPPER_RUNTIME_MIN - PBL8_GRIPPER_SATURATION_MARGIN
        <= value
        <= PBL8_GRIPPER_RUNTIME_MAX + PBL8_GRIPPER_SATURATION_MARGIN
    )


def apply_pbl8_gripper_saturation(action):
    """Copy an action and saturate only its normalized gripper field."""
    applied = list(action)
    raw_gripper = float(applied[6])
    if not is_within_pbl8_gripper_saturation_margin(raw_gripper):
        raise ValueError(
            "PBL8 gripper outside saturation margin "
            "[-1.010000, 1.010000]: %.9f" % raw_gripper
        )
    applied[6] = min(
        PBL8_GRIPPER_RUNTIME_MAX,
        max(PBL8_GRIPPER_RUNTIME_MIN, raw_gripper),
    )
    return applied
