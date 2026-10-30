"""Runtime wire/model image keys shared by observation adapters."""
from piper_vla_dual_runtime.core import CAMERAS

STATE_KEY = "observation/state"
IMAGE_KEYS = {name: "observation/" + name + "_image" for name in CAMERAS}
MODEL_IMAGES = dict(zip(CAMERAS, ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")))
