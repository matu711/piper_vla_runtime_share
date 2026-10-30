"""Read-only training packages, shared by single/dual and pi0/pi05 policies.

No package Python is executed. Supported joint contracts use the existing
Libero (single) / Aloha(False) (dual) transforms. Unknown contracts fail closed.
Heavy OpenPI imports are deferred so inspection/launch works on the host.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

import yaml


TOKENIZER_SHA256 = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
# Shared inference profiles, based on Pi0Config + the registered PiPER joint
# transforms. Seven combinations have config/transform tests only; the real
# bimanual/pi05/LoRA package is additionally GPU verified. Never an EE fallback.
MODEL_PROFILES = {
    "pi0": dict(pi05=False, max_token_len=48, discrete_state_input=False, normalization="standard"),
    "pi05": dict(pi05=True, max_token_len=200, discrete_state_input=True, normalization="quantile"),
}
ROBOT_PROFILES = {
    "single": dict(width=7, cameras=["front", "wrist"], layout="single7", adapter="LiberoInputs"),
    "bimanual": dict(width=14, cameras=["front", "left_wrist", "right_wrist"], layout="left7_right7", adapter="AlohaInputs(False)"),
}
JOINT_FORMATS = {"delta_joint_absolute_gripper": True, "absolute_joint": False, "joint_absolute": False}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inside(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("package path must be relative: " + str(relative))
    path = (root / relative).resolve(strict=True)
    if root not in path.parents:
        raise ValueError("package path escapes root: " + str(relative))
    return path


def detect_format(root):
    root = Path(root)
    bundle = (root / "manifest.yaml").is_file()
    package = (root / "metadata.json").is_file()
    if bundle and package:
        raise ValueError("ambiguous model folder: manifest.yaml is not a supported model input")
    if bundle:
        raise ValueError("legacy Bundle input is no longer supported; use a training Checkpoint package")
    if package:
        return "training_package"
    raise ValueError("training Checkpoint package requires metadata.json")


def verify_manifest(root):
    manifest = json.loads(inside(root, "MANIFEST.sha256.json").read_text())
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("invalid SHA256 manifest")
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
              and p.name != "MANIFEST.sha256.json"}
    if actual != set(manifest):
        raise ValueError("SHA256 manifest file set mismatch: " + str(actual ^ set(manifest)))
    for name, digest in manifest.items():
        if not isinstance(digest, str) or len(digest) != 64 or sha256(inside(root, name)) != digest:
            raise ValueError("SHA256 mismatch: " + name)
    return manifest


def _agree(name, *values):
    present = [v for v in values if v is not None]
    if not present or any(v != present[0] for v in present):
        raise ValueError("missing or conflicting " + name + ": " + repr(present))
    return present[0]


def _positive(name, value):
    if type(value) is not int or value < 1:
        raise ValueError(name + " must be a positive integer")
    return value


def parameter_shapes(checkpoint):
    """Global array shape metadata; write_shape in _METADATA may be sharded."""
    raw = json.loads((checkpoint / "params/_METADATA").read_text())
    tree = raw["tree_metadata"]
    paths = {".".join(ast.literal_eval(k)) for k in tree}
    shapes = {}
    for path in sorted((checkpoint / "params/array_metadatas").glob("*")):
        for item in json.loads(path.read_text())["array_metadatas"]:
            item = item["array_metadata"]
            name, shape = item["param_name"], item["write_shape"]
            if name in shapes and shapes[name] != shape:
                raise ValueError("inconsistent global shape: " + name)
            shapes[name] = shape
    if not shapes or set(shapes) != paths:
        raise ValueError("Orbax array metadata incomplete")
    return {k.removeprefix("params.").removesuffix(".value"): v for k, v in shapes.items()}


def inspect_package(folder, *, verify=True):
    root = Path(folder).resolve(strict=True)
    if detect_format(root) != "training_package":
        raise ValueError("expected training package")
    manifest = verify_manifest(root) if verify else None
    metadata = json.loads(inside(root, "metadata.json").read_text())
    training = yaml.safe_load(inside(root, "training.yaml").read_text())
    tuning = json.loads(inside(root, "tuning_norm_config.json").read_text())
    robot = _agree("robot", metadata.get("robot"), training.get("robot"), tuning.get("robot"))
    family = _agree("model", metadata.get("model"), training.get("model"), tuning.get("model"))
    method = _agree("method", metadata.get("method"), training.get("method"))
    if robot not in ("single", "bimanual") or family not in ("pi0", "pi05") or method not in ("full", "lora"):
        raise ValueError("unsupported robot/model/method")
    if "use_lora" in training and training["use_lora"] is not (method == "lora"):
        raise ValueError("use_lora/method mismatch")
    action_format = _agree("action_format", training.get("action_format"), tuning.get("action_format"), metadata.get("action_format"))
    if action_format not in JOINT_FORMATS:
        raise ValueError("unsupported action contract (EE requires its exact registered adapter): " + action_format)
    horizon = _positive("action_horizon", _agree("action_horizon", training.get("action_horizon"), tuning.get("action_horizon"), metadata.get("action_horizon")))
    repo = _agree("dataset_repo_id", training.get("data_collection", {}).get("dataset_repo_id"), tuning.get("repo_id"), metadata.get("dataset_repo_id"))
    asset_values = [s["asset_id"] for s in (metadata, training, tuning) if s.get("asset_id") is not None]
    asset_id = _agree("asset_id", *asset_values) if asset_values else repo
    candidates = sorted(p.name for p in inside(root, "checkpoint").iterdir() if p.is_dir() and p.name.isdigit())
    selected = metadata.get("checkpoint_step", metadata.get("step"))
    if selected is None:
        if len(candidates) != 1:
            raise ValueError("ambiguous checkpoint step; metadata must select one")
        selected = candidates[0]
    if str(selected) not in candidates:
        raise ValueError("selected checkpoint step missing")
    checkpoint = inside(root, "checkpoint/" + str(selected))
    ckmeta = json.loads(inside(root, str(checkpoint.relative_to(root)) + "/_CHECKPOINT_METADATA").read_text())
    if "params" not in ckmeta.get("item_handlers", {}):
        raise ValueError("checkpoint metadata has no params handler")
    for filename in ("params/_METADATA", "params/manifest.ocdbt", "params/_sharding"):
        inside(root, str(checkpoint.relative_to(root)) + "/" + filename)
    stats_file = inside(root, str(checkpoint.relative_to(root)) + "/assets/" + asset_id + "/norm_stats.json")
    if json.loads(stats_file.read_text()) != json.loads(inside(root, "norm_stats.json").read_text()):
        raise ValueError("root/checkpoint normalization statistics mismatch")
    stats = json.loads(stats_file.read_text())["norm_stats"]
    robot_profile, model_profile = ROBOT_PROFILES[robot], MODEL_PROFILES[family]
    width = robot_profile["width"]
    declared_norms = [s["normalization"] for s in (metadata, training, tuning) if "normalization" in s]
    declared_norms += ["quantile" if s["use_quantile_norm"] else "standard" for s in (metadata, training, tuning)
                       if "use_quantile_norm" in s and type(s["use_quantile_norm"]) is bool]
    normalization = _agree("normalization", *declared_norms) if declared_norms else model_profile["normalization"]
    if normalization not in ("standard", "quantile"):
        raise ValueError("unsupported normalization")
    if set(stats) != {"state", "actions"}:
        raise ValueError("unsupported normalization keys")
    for source in (training, tuning, metadata):
        if source.get("normalization", normalization) != normalization or source.get("standard_norm_keys", []):
            raise ValueError("unsupported or conflicting normalization")
        if source.get("use_quantile_norm", normalization == "quantile") is not (normalization == "quantile"):
            raise ValueError("use_quantile_norm mismatch")
        for name in ("state_dim", "action_dim"):
            if source.get(name, width) != width:
                raise ValueError("external " + name + " mismatch")
        if source.get("adapt_to_pi", False) is not False:
            raise ValueError("joint PiPER package requires adapt_to_pi=False")
    for key in ("state", "actions"):
        entry = stats[key]
        for name in (("mean", "std", "q01", "q99") if normalization == "quantile" else ("mean", "std")):
            values = entry.get(name)
            if not isinstance(values, list) or len(values) != width or not all(type(v) in (int, float) and math.isfinite(v) for v in values):
                raise ValueError("invalid stats " + key + "." + name)
        if any(v < 0 for v in entry["std"]) or (normalization == "quantile" and any(a > b for a, b in zip(entry["q01"], entry["q99"]))):
            raise ValueError("invalid normalization bounds")
    shapes = parameter_shapes(checkpoint)
    internal = shapes["action_out_proj.bias"][0]
    if internal < width or shapes["action_in_proj.kernel"][0] != internal:
        raise ValueError("internal action projection mismatch")
    ranks = []
    variants = []
    for suffix, variant in (("", "gemma_2b"), ("_1", "gemma_300m")):
        prefix = "PaliGemma.llm.layers.mlp" + suffix + "."
        lora = shapes.get(prefix + "linear_lora_a")
        rank = lora[-1] if lora else None
        ranks.append(rank)
        variants.append(variant + ("_lora" if rank is not None else ""))
    if (method == "full" and any(ranks)) or (method == "lora" and not any(ranks)):
        raise ValueError("Full/LoRA parameter structure mismatch")
    rank_override = _agree("lora_rank", training.get("lora_rank"), metadata.get("lora_rank")) if any(
        v is not None for v in (training.get("lora_rank"), metadata.get("lora_rank"))) else None
    if rank_override is not None:
        _positive("lora_rank", rank_override)
    expected_ranks = [rank_override or 16, rank_override or 32]
    if any(rank is not None and rank != expected for rank, expected in zip(ranks, expected_ranks)):
        raise ValueError("LoRA rank does not match Pi0Config/variants")
    for source in (metadata, training):
        for key, expected in (("language_lora_rank", ranks[0]), ("action_expert_lora_rank", ranks[1])):
            if key in source and source[key] != expected:
                raise ValueError(key + " mismatch")
    pi05 = model_profile["pi05"]
    if ("state_proj.kernel" in shapes) == pi05:
        raise ValueError("pi0/pi05 parameter structure mismatch")
    # These defaults are the actual vendored Pi0Config defaults, not dimensions
    # copied from the current checkpoint. Explicit settings are cross-checked.
    model = dict(pi05=pi05, action_dim=internal, action_horizon=horizon,
                 paligemma_variant=variants[0], action_expert_variant=variants[1],
                 lora_rank=rank_override, dtype="bfloat16", max_token_len=model_profile["max_token_len"],
                 discrete_state_input=model_profile["discrete_state_input"])
    explicit = {}
    for source in (metadata, training, tuning):
        settings = dict(source.get("model_config", {}))
        if "internal_action_dim" in source:
            settings["action_dim"] = source["internal_action_dim"]
        for key in model:
            if key in source and key != "action_dim":
                if key in settings and settings[key] != source[key]:
                    raise ValueError("conflicting model setting: " + key)
                settings[key] = source[key]
        for key, value in settings.items():
            if key in explicit and explicit[key] != value:
                raise ValueError("conflicting model setting: " + key)
            explicit[key] = value
    for key, value in explicit.items():
        if key not in model:
            raise ValueError("unsupported model setting: " + key)
        if key in ("max_token_len", "dtype", "discrete_state_input"):
            model[key] = value
        elif model[key] != value:
            raise ValueError("model/parameter mismatch: " + key)
    if model["dtype"] not in ("bfloat16", "float32") or model["discrete_state_input"] is not pi05:
        raise ValueError("unsupported model dtype/state tokenization")
    _positive("max_token_len", model["max_token_len"])
    # Only the declared joint contracts are mapped to the established adapters.
    cameras = robot_profile["cameras"]
    camera_map = ({"cam_high": "observation.images.front", "cam_left_wrist": "observation.images.left_wrist",
                   "cam_right_wrist": "observation.images.right_wrist"} if robot == "bimanual" else
                  {"observation/image": "observation.images.front", "observation/wrist_image": "observation.images.wrist"})
    for source in (training, metadata, tuning):
        if source.get("cameras", cameras) != cameras:
            raise ValueError("unsupported camera contract; exact adapter required")
        if source.get("camera_map", camera_map) != camera_map:
            raise ValueError("camera mapping mismatch")
        for key, expected in (("joint_unit", "rad"), ("gripper_unit", "m"), ("layout", robot_profile["layout"])):
            if source.get(key, expected) != expected:
                raise ValueError("physical contract mismatch: " + key)
    delta = JOINT_FORMATS[action_format]
    mask = ([True] * 6 + [False]) * (2 if robot == "bimanual" else 1)
    for source in (training, metadata, tuning):
        if "delta_mask" in source and source["delta_mask"] != (mask if delta else []):
            raise ValueError("joint delta mask mismatch")
    contract = dict(action_type="joint_absolute", state_dim=width, action_dim=width,
                    action_horizon=horizon, internal_action_dim=internal,
                    normalization=normalization, joint_unit="rad", gripper_unit="m",
                    layout=robot_profile["layout"], cameras=cameras)
    result = dict(root=str(root), checkpoint=str(checkpoint), config_name=metadata["config"],
                  robot=robot, family=family, method=method, model=model, ranks=ranks,
                  repo_id=repo, asset_id=asset_id, stats_path=str(stats_file), contract=contract,
                  delta_mask=mask if delta else [], action_format=action_format,
                  stats=stats, step=int(selected), parameter_shapes=shapes,
                  checksums_verified=manifest is not None,
                  settings_source="training.yaml + metadata.json + Orbax; vendored Pi0Config/joint adapters")
    return result


def build_train_config(spec, tokenizer_path):
    from openpi import transforms
    from openpi.models.pi0_config import Pi0Config
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.policies import aloha_policy, libero_policy
    from openpi.training import config
    import sentencepiece

    tokenizer_path = Path(tokenizer_path)
    if sha256(tokenizer_path) != TOKENIZER_SHA256:
        raise ValueError("unsupported PaliGemma tokenizer SHA256")

    class LocalTokenizer(PaligemmaTokenizer):
        def __init__(self, max_len):
            self._max_len = max_len
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=tokenizer_path.read_bytes())

    @dataclasses.dataclass(frozen=True)
    class PackageDataConfig(config.DataConfigFactory):
        def create(self, assets_dirs, model_config):
            dual = spec["robot"] == "bimanual"
            inputs = [aloha_policy.AlohaInputs(adapt_to_pi=False)] if dual else [libero_policy.LiberoInputs(model_config.model_type)]
            outputs = [aloha_policy.AlohaOutputs(adapt_to_pi=False)] if dual else [libero_policy.LiberoOutputs()]
            if spec["delta_mask"]:
                inputs.append(transforms.DeltaActions(tuple(spec["delta_mask"])))
                outputs.insert(0, transforms.AbsoluteActions(tuple(spec["delta_mask"])))
            return dataclasses.replace(self.create_base_config(assets_dirs, model_config),
                use_quantile_norm=spec["contract"]["normalization"] == "quantile",
                data_transforms=transforms.Group(inputs=inputs, outputs=outputs),
                model_transforms=transforms.Group(inputs=[
                    transforms.InjectDefaultPrompt(None), transforms.ResizeImages(224, 224),
                    transforms.TokenizePrompt(LocalTokenizer(model_config.max_token_len), discrete_state_input=model_config.discrete_state_input),
                    transforms.PadStatesAndActions(model_config.action_dim)]))

    model = Pi0Config(**spec["model"])
    data = PackageDataConfig(repo_id=spec["repo_id"], assets=config.AssetsConfig(
        assets_dir=str(Path(spec["checkpoint"]) / "assets"), asset_id=spec["asset_id"]))
    return config.TrainConfig(name=spec["config_name"], exp_name="package_inference", model=model,
                             data=data, ema_decay=None, wandb_enabled=False)


class PackagePolicy:
    def __init__(self, policy, spec, train_config, norm_stats):
        self.policy, self.spec, self.train_config = policy, spec, train_config
        self.norm_stats = norm_stats
        self.data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        self.metadata = dict(policy.metadata, **spec["contract"])
        self.manifest = runtime_manifest(spec)

    def adapt(self, observation):
        if self.spec["robot"] == "bimanual":
            from piper_vla_dual_runtime.observation_adapter import adapt_runtime_observation
            return adapt_runtime_observation(observation)
        from piper_vla_common.adapters import AdapterProfile
        adapted = AdapterProfile("observation/state", {"front": "observation/image", "wrist": "observation/wrist_image"}, state_width=7).adapt(observation)
        return {"observation/state": adapted["state"], "prompt": adapted["prompt"],
                "observation/image": adapted["images"]["front"], "observation/wrist_image": adapted["images"]["wrist"]}

    def infer(self, observation):
        import numpy as np
        result = self.policy.infer(self.adapt(observation))
        actions = np.asarray(result["actions"])
        expected = (self.spec["model"]["action_horizon"], self.spec["contract"]["action_dim"])
        if actions.shape != expected or not np.isfinite(actions).all():
            raise ValueError("invalid final physical Action; expected " + str(expected))
        return result

    def warm_up(self, prompt=None):
        import time
        import numpy as np
        prompt = "Runtime compatibility warm-up." if prompt is None else prompt
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 2048 or "\x00" in prompt:
            raise ValueError("invalid warm-up prompt")
        keys = (["observation/front_image", "observation/left_wrist_image", "observation/right_wrist_image"]
                if self.spec["robot"] == "bimanual" else ["observation/image", "observation/wrist_image"])
        observation = {"observation/state": np.asarray(self.norm_stats["state"].mean, dtype=np.float32),
                       "prompt": prompt, **{key: np.zeros((224, 224, 3), dtype=np.uint8) for key in keys}}
        start = time.perf_counter()
        self.infer(observation)
        return (time.perf_counter() - start) * 1000


def load_policy(folder, *, tokenizer_path=None, expected_robot=None):
    """Single model-loading entry for all eight training-package profiles."""
    spec = inspect_package(folder)
    if expected_robot is not None and spec["robot"] != expected_robot:
        raise ValueError("wrong Robot for launch path")
    tokenizer_path = tokenizer_path or os.environ.get("PIPER_TOKENIZER_PATH") or (
        Path(__file__).resolve().parents[2] / "assets/paligemma_tokenizer.model")
    train_config = build_train_config(spec, tokenizer_path)
    # Check ALL names/shapes before loading: OpenPI otherwise removes extra
    # parameters by intersecting trees (could hide a Full/LoRA mismatch).
    import flax.nnx as nnx
    import jax
    from flax import traverse_util
    expected = nnx.state(nnx.eval_shape(train_config.model.create, jax.random.key(0))).to_pure_dict()
    shapes = {".".join(path): list(value.shape) for path, value in traverse_util.flatten_dict(expected).items()}
    if shapes != spec["parameter_shapes"]:
        missing = set(shapes) - set(spec["parameter_shapes"])
        extra = set(spec["parameter_shapes"]) - set(shapes)
        wrong = [k for k in shapes.keys() & spec["parameter_shapes"].keys() if shapes[k] != spec["parameter_shapes"][k]]
        raise ValueError(f"parameter tree mismatch: missing={missing}, extra={extra}, wrong_shape={wrong}")
    from openpi.shared import normalize
    from openpi.policies import policy_config
    stats = normalize.load(Path(spec["stats_path"]).parent)
    policy = policy_config.create_trained_policy(train_config, spec["checkpoint"], norm_stats=stats)
    return PackagePolicy(policy, spec, train_config, stats)


def runtime_manifest(spec):
    result = dict(model_family=spec["family"], config_name=spec["config_name"],
                  dataset_repo_id=spec["repo_id"], action_type="joint_absolute",
                  action_horizon=spec["model"]["action_horizon"], contract=spec["contract"],
                  observation=dict(cameras=spec["contract"]["cameras"], state=dict(shape=[spec["contract"]["state_dim"]])))
    if spec["robot"] == "bimanual":
        # Existing dual ROS Runtime supports only this physical wire contract.
        # Offline Policy inference itself supports other horizons/dimensions.
        if spec["model"]["action_horizon"] != 50 or spec["model"]["action_dim"] != 32:
            result["runtime_unsupported"] = "dual ROS requires horizon50/internal32"
        result["compatibility_profile"] = "dual_pi05_joint_v1" if spec["family"] == "pi05" else "dual_pi0_joint_v1"
        delta = bool(spec["delta_mask"])
        result["training_actions"] = dict(joint_delta_indices=[i for i, v in enumerate(spec["delta_mask"]) if v],
            absolute_gripper_indices=[6, 13], delta_reference="request_state" if delta else None,
            output_transforms=["Unnormalize"] + (["AbsoluteActions"] if delta else []) + ["truncate_to_14"])
        if spec["family"] == "pi05" and not delta:
            result["runtime_unsupported"] = "dual pi05 ROS profile requires joint deltas"
    return result


def launch(folder, robot, runtime_dir, dry_run=True, prompt=None, max_steps=None, offline=False):
    """New-format launch only; generated files live outside the package."""
    root, runtime_dir = Path(folder).resolve(strict=True), Path(runtime_dir).resolve(strict=True)
    if max_steps is not None and (type(max_steps) is not int or max_steps < 0):
        raise ValueError("max-execution-steps must be a nonnegative integer")
    spec = inspect_package(root)
    if spec["robot"] != robot:
        raise ValueError("package Robot does not match launch script")
    token = os.environ.get("PIPER_TOKENIZER_PATH")
    if not token:
        # Reuse an already installed, verified tokenizer. Never download one.
        candidates = [runtime_dir / "assets/paligemma_tokenizer.model"]
        candidates = [p for p in candidates if p.is_file()]
        token = next((str(p) for p in candidates if sha256(p) == TOKENIZER_SHA256), None)
    if not token or sha256(token) != TOKENIZER_SHA256:
        raise ValueError("verified local tokenizer missing; set PIPER_TOKENIZER_PATH once")
    output = Path(tempfile.mkdtemp(prefix="piper-package-"))
    manifest = runtime_manifest(spec)
    (output / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    # A diagnostic prompt is only for synthetic warm-up, never the real task.
    # Runtime Start still requires the user's task prompt.
    prompt = prompt or "Runtime compatibility warm-up."
    mounts = ["-v", str(root) + ":/package:ro", "-v", str(runtime_dir / "src") + ":/runtime/src:ro",
              "-v", str(Path(token).resolve()) + ":/tokenizer.model:ro", "-v", str(output) + ":/results"]
    image = os.environ.get("PIPER_PACKAGE_IMAGE", "piper-vla-dual-policy:local")
    if offline:
        command = ["docker", "run", "--gpus", "all", "--network", "none",
                   "-e", "PYTHONPATH=/runtime/src", "-e", "PYTHONDONTWRITEBYTECODE=1",
                   "-e", "PIPER_TOKENIZER_PATH=/tokenizer.model", *mounts,
                   "-v", str(runtime_dir / "scripts/verify_checkpoint_package.py") + ":/verify.py:ro",
                   "--entrypoint", "/opt/openpi/.venv/bin/python", image, "/verify.py",
                   "--package", "/package", "--prompt", prompt, "--output", "/results/gpu.json"]
        subprocess.run(command, check=True)
        print("Offline report: " + str(output / "gpu.json"))
        return
    if "runtime_unsupported" in manifest:
        raise ValueError(manifest["runtime_unsupported"] + "; use --offline")
    config = yaml.safe_load((runtime_dir / "configs" / ("runtime.dual.yaml" if robot == "bimanual" else "runtime.yaml")).read_text())
    config["model"] = dict(checkpoint="/package", manifest="/package-runtime/manifest.yaml")
    config["runtime"]["dry_run"] = dry_run
    config["policy"]["action_horizon"] = spec["model"]["action_horizon"]
    if max_steps is not None:
        config["runtime"]["max_execution_steps"] = int(max_steps)
    if config["runtime"]["execution_chunk_length"] > spec["model"]["action_horizon"]:
        raise ValueError("runtime execution chunk exceeds model horizon; use --offline")
    (output / "runtime.yaml").write_text(yaml.safe_dump(config))
    port = 8001 if robot == "bimanual" else 8000
    client_image = os.environ.get("PIPER_PACKAGE_CLIENT_IMAGE", "piper-vla-dual-client:local")
    # Reuse the existing client image; offline operation needs only policy image.
    services = {"policy": dict(image=image, gpus="all", ports=[f"{port}:{port}"],
        environment=dict(PYTHONPATH="/runtime/src", PYTHONDONTWRITEBYTECODE="1", PIPER_TOKENIZER_PATH="/tokenizer.model"),
        volumes=[str(root)+":/package:ro",str(runtime_dir/"src")+":/runtime/src:ro",str(Path(token).resolve())+":/tokenizer.model:ro",
                 str(runtime_dir/"docker/serve_checkpoint_package.py")+":/serve-package.py:ro"],
        entrypoint=["/opt/openpi/.venv/bin/python", "/serve-package.py"],
        command=["--package", "/package", "--robot", robot, "--warmup-prompt", prompt, "--port", str(port)]),
        "client": dict(image=client_image, network_mode="host", command=["python3", "/runtime/src/" + (
            "piper_vla_dual_runtime" if robot == "bimanual" else "piper_vla_runtime") + "/node.py"], environment=dict(
            POLICY_HOST="127.0.0.1", POLICY_PORT=str(port), RUNTIME_CONFIG="/package-runtime/runtime.yaml",
            ROS_MASTER_URI=os.environ.get("ROS_MASTER_URI", "http://127.0.0.1:11311"), ROS_IP=os.environ.get("ROS_IP", "127.0.0.1")),
            volumes=[str(root)+":/package:ro",str(output)+":/package-runtime:ro",str(runtime_dir/"src")+":/runtime/src:ro",
                     str(output/"logs")+":/logs"]) }
    (output / "logs").mkdir()
    compose = output / "compose.yaml"
    project = "piper-vla-dual" if robot == "bimanual" else "piper-vla"
    compose.write_text(yaml.safe_dump(dict(name=project, services=services)))
    command = ["docker", "compose", "--env-file", "/dev/null", "-p", project, "-f", str(compose)]
    subprocess.run(command + ["up", "-d", "--no-build", "--pull", "never", "policy"], check=True)
    # Existing readiness check uses only the Python standard library.
    subprocess.run(["python3", str(runtime_dir/"scripts/wait_policy_ready.py"), "--host", "127.0.0.1", "--port", str(port), "--attempts", "120", "--interval", "1"], check=True)
    subprocess.run(command + ["up", "-d", "--no-build", "--pull", "never", "--force-recreate", "client"], check=True)
    print("READY (Start with prompt required); generated config: " + str(output))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", "--checkpoint", dest="package", required=True)
    parser.add_argument("--robot", choices=("single", "bimanual"), required=True)
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--warmup-prompt")
    parser.add_argument("--controller-run", action="store_true")
    parser.add_argument("--dry-run", dest="controller_run", action="store_false")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--max-execution-steps", type=int)
    args = parser.parse_args()
    launch(args.package, args.robot, args.runtime_dir, not args.controller_run,
           args.warmup_prompt, args.max_execution_steps, args.offline)


if __name__ == "__main__":
    main()
