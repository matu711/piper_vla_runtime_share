"""Runtime wire format -> frozen bimanual AlohaInputs format, exactly once."""
from piper_vla_common.adapters import AdapterProfile

from piper_vla_dual_runtime.policy_transforms import STATE_KEY, IMAGE_KEYS

TRAINING_CAMERAS = {
    "front": "cam_high", "left_wrist": "cam_left_wrist", "right_wrist": "cam_right_wrist",
}


PROFILE = AdapterProfile(STATE_KEY,
    {training_name: IMAGE_KEYS[camera] for camera, training_name in TRAINING_CAMERAS.items()},
    state_width=14, chw=True)


def adapt_runtime_observation(observation):
    return PROFILE.adapt(observation)
