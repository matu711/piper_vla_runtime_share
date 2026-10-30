#!/usr/bin/env python3
"""Offline real-weight verification. No ROS, cameras, CAN or controller access."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from piper_vla_common.checkpoint_package import load_policy


def observation(policy, prompt):
    state = np.asarray(policy.norm_stats["state"].mean, dtype=np.float32)
    pixels = np.arange(224 * 224 * 3, dtype=np.uint32).reshape(224, 224, 3)
    cameras = ("front", "left_wrist", "right_wrist") if policy.spec["robot"] == "bimanual" else ("front", "wrist")
    keys = ["observation/" + name + "_image" for name in cameras]
    if policy.spec["robot"] == "single":
        keys = ["observation/image", "observation/wrist_image"]
    return {"observation/state": state, "prompt": prompt,
            **{key: ((pixels + i * 31) % 256).astype(np.uint8) for i, key in enumerate(keys)}}


def transform_roundtrip(policy, obs):
    """Verify joint order, delta reference and absolute grippers numerically."""
    from openpi import transforms
    original = policy.adapt(obs)
    horizon, width = policy.spec["contract"]["action_horizon"], policy.spec["contract"]["action_dim"]
    actions = np.tile(obs["observation/state"], (horizon, 1)).astype(np.float64)
    actions += np.arange(1, width + 1)[None, :] * .001
    actions[:, 6::7] = np.arange(1, width // 7 + 1)[None, :] * .02
    data = policy.data_config
    encoded = transforms.compose([*data.data_transforms.inputs,
        transforms.Normalize(policy.norm_stats, use_quantiles=data.use_quantile_norm),
        *data.model_transforms.inputs])(dict(original, actions=actions.copy()))
    decoded = transforms.compose([transforms.Unnormalize(policy.norm_stats, use_quantiles=data.use_quantile_norm),
                                  *data.data_transforms.outputs])({"state": encoded["state"], "actions": encoded["actions"]})
    np.testing.assert_allclose(decoded["actions"], actions, rtol=1e-5, atol=2e-6)
    # Shift only the request State after normalization: only delta joints may
    # shift in the restored output. Grippers must remain unchanged.
    reference_state = obs["observation/state"].copy()
    shifted = dict(state=reference_state, actions=actions.copy())
    for transform in data.data_transforms.inputs:
        shifted = transform(shifted) if not type(transform).__name__.endswith("Inputs") else shifted
    delta_mask = np.asarray(policy.spec["delta_mask"], dtype=bool)
    if delta_mask.size:
        np.testing.assert_allclose(shifted["actions"][:, delta_mask],
            actions[:, delta_mask] - reference_state[delta_mask], atol=1e-7)
        np.testing.assert_allclose(shifted["actions"][:, ~delta_mask], actions[:, ~delta_mask], atol=1e-7)
    return dict(passed=True, max_absolute_error=float(np.max(np.abs(decoded["actions"] - actions))),
        internal_state_shape=list(encoded["state"].shape), internal_action_shape=list(encoded["actions"].shape),
        inputs=[type(t).__name__ for t in data.data_transforms.inputs],
        outputs=["Unnormalize"]+[type(t).__name__ for t in data.data_transforms.outputs],
        image_masks={k: bool(v) for k, v in encoded["image_mask"].items()},
        delta_mask=policy.spec["delta_mask"], units=policy.spec["contract"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = dict(success=False, software_only=True, package=str(args.package))
    try:
        import jax
        if jax.default_backend() != "gpu":
            raise RuntimeError("GPU backend required")
        report.update(jax_version=jax.__version__, devices=[str(d) for d in jax.devices()])
        report["gpu_before"] = subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"], text=True).strip()
        print("verifying hashes, settings, parameter shapes and restoring weights", flush=True)
        start = time.perf_counter()
        policy = load_policy(args.package)
        report.update(load_and_verify_seconds=time.perf_counter()-start,
            checkpoint_loaded=True, parameter_arrays=len(policy.spec["parameter_shapes"]),
            model=policy.spec["model"], ranks=policy.spec["ranks"], contract=policy.spec["contract"],
            checksums_verified=policy.spec["checksums_verified"])
        print("checkpoint restored", flush=True)
        obs = observation(policy, args.prompt)
        report["transform_roundtrip"] = transform_roundtrip(policy, obs)
        report["inference"] = []
        for i in range(3):
            start = time.perf_counter()
            result = policy.infer(obs)
            actions = np.asarray(result["actions"])
            entry = dict(run=i, wall_ms=(time.perf_counter()-start)*1000,
                shape=list(actions.shape), finite=bool(np.isfinite(actions).all()),
                first_action=actions[0].tolist(), min=float(actions.min()), max=float(actions.max()))
            report["inference"].append(entry)
            print(json.dumps(entry), flush=True)
        report["jax_memory_bytes"] = jax.devices()[0].memory_stats()
        report["gpu_memory_MiB"] = subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"], text=True).strip()
        report["success"] = True
    except Exception as exc:
        report["error"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
