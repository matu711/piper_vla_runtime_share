"""Physical image/state validation and layout; OpenPI owns learned transforms."""
from dataclasses import dataclass

import numpy as np
from openpi_client import image_tools


def state_vector(values, width=None, floating=False):
    value = np.asarray(values)
    if (value.ndim != 1 or (width is not None and value.shape != (width,))
            or not np.issubdtype(value.dtype, np.number)
            or (floating and not np.issubdtype(value.dtype, np.floating))
            or not np.isfinite(value).all()):
        raise ValueError("expected finite state vector[%s]" % width)
    return value.copy()


def rgb_image(value, chw=False):
    value = np.asarray(value)
    if (value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3
            or min(value.shape[:2]) < 1):
        raise ValueError("expected uint8 RGB[H,W,3]")
    return np.ascontiguousarray(value.transpose(2, 0, 1)) if chw else value


def prepare_image(rgb, size=224, rotation=0):
    # Preserve the existing client resize implementation and order of operations.
    value = rgb_image(rgb)
    if rotation:
        value = np.rot90(value, k=rotation).copy()
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(value, size, size))


@dataclass(frozen=True)
class AdapterProfile:
    state_key: str
    image_keys: object
    state_width: int = None
    chw: bool = False

    def adapt(self, observation):
        required = (self.state_key, "prompt", *self.image_keys.values())
        missing = [key for key in required if key not in observation]
        if missing:
            raise ValueError("missing Runtime observation keys: " + ", ".join(missing))
        if not isinstance(observation["prompt"], str):
            raise ValueError("prompt must be a string")
        return {
            "state": state_vector(observation[self.state_key], self.state_width, floating=True),
            "images": {name: rgb_image(observation[key], self.chw)
                       for name, key in self.image_keys.items()},
            "prompt": observation["prompt"],
        }
