"""Single-arm physical State -> existing Libero wire profile, including PBL8."""
import numpy as np

from piper_vla_common.adapters import AdapterProfile, state_vector


def policy_state(message, manifest, is_pbl8, pbl8_config):
    values = state_vector(np.asarray(message.position, dtype=np.float32))
    expected = manifest.get("observation", {}).get("state", {}).get("shape")
    if is_pbl8:
        if len(values) < 7:
            raise ValueError("PBL8 requires /joint_states_single [j1..j6, gripper_width_m]")
        if expected == [7]:
            values = values[:7]
        else:
            open_width = float(pbl8_config["gripper_open_width_m"])
            qmax = float(pbl8_config["finger_qpos_max_m"])
            if not np.isfinite([open_width, qmax]).all() or open_width <= 0 or qmax <= 0:
                raise ValueError("invalid PBL8 gripper state conversion")
            finger = np.clip(float(values[6]) / open_width, 0.0, 1.0) * qmax
            values = np.concatenate([values[:6], np.asarray([finger, finger], dtype=np.float32)])
    if expected and list(values.shape) != list(expected):
        raise ValueError("policy state shape mismatch: expected %s, got %s" % (expected, list(values.shape)))
    return values


def build_observation(snapshot, config, is_pbl8, state_builder, image_builder):
    """Keep existing Libero wire keys; its OpenPI transform owns image masks."""
    obs_cfg, pbl8_cfg = config["observation"], config.get("pbl8", {})
    observation = {
        obs_cfg["state_key"]: state_builder(snapshot.messages["state"]),
        obs_cfg["front_key"]: image_builder(snapshot.messages["front"]),
        obs_cfg["wrist_key"]: image_builder(snapshot.messages["wrist"], rotate_left_90=(
            is_pbl8 and bool(pbl8_cfg.get("wrist_rotate_left_90", True)))),
        "prompt": snapshot.prompt,
    }
    profile = AdapterProfile(obs_cfg["state_key"],
                             {"front": obs_cfg["front_key"], "wrist": obs_cfg["wrist_key"]})
    adapted = profile.adapt(observation)
    return {obs_cfg["state_key"]: adapted["state"],
            **{profile.image_keys[name]: value for name, value in adapted["images"].items()},
            "prompt": adapted["prompt"]}
