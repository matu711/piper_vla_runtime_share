"""Eight joint profiles: real OpenPI transforms, fake model/Orbax payloads.

Run in the existing policy Python environment. These tests are NOT GPU inference.
"""
import copy
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import yaml

from piper_vla_common import checkpoint_package as package

ROOT = Path(__file__).resolve().parents[2]
TOKENIZER = os.environ.get("PIPER_TOKENIZER_PATH", "/tokenizer.model")


def fixture(root, robot="bimanual", family="pi05", method="lora", horizon=50, delta=True, internal=32):
    step = root / "checkpoint/39999"
    (step / "params/array_metadatas").mkdir(parents=True)
    repo = "local/test"
    stats_dir = step / "assets" / repo
    stats_dir.mkdir(parents=True)
    width = 14 if robot == "bimanual" else 7
    stats = {key: dict(mean=[.1]*width, std=[.5]*width, q01=[-.5]*width, q99=[.7]*width) for key in ("state", "actions")}
    for path in (root/"norm_stats.json", stats_dir/"norm_stats.json"):
        path.write_text(json.dumps(dict(norm_stats=stats)))
    meta = dict(robot=robot, model=family, method=method, config=f"{family}_piper_{robot}_{method}")
    training = dict(robot=robot, model=family, method=method, action_horizon=horizon,
        action_format="delta_joint_absolute_gripper" if delta else "absolute_joint", lora_rank=None,
        use_lora=method=="lora", data_collection=dict(dataset_repo_id=repo))
    tuning = {**{k: training[k] for k in ("robot", "model", "action_horizon", "action_format")}, "repo_id": repo}
    (root/"metadata.json").write_text(json.dumps(meta))
    (root/"training.yaml").write_text(yaml.safe_dump(training))
    (root/"tuning_norm_config.json").write_text(json.dumps(tuning))
    (step/"_CHECKPOINT_METADATA").write_text(json.dumps(dict(item_handlers=dict(params="orbax"))))
    shapes = {"action_out_proj.bias": [internal], "action_in_proj.kernel": [internal,1024]}
    if family == "pi0":
        shapes["state_proj.kernel"] = [internal,1024]
    if method == "lora":
        for suffix, rank in (("",16),("_1",32)):
            shapes[f"PaliGemma.llm.layers.mlp{suffix}.linear_lora_a"] = [18,4096,rank]
    tree = {str(tuple(["params"]+k.split(".")+["value"])): {} for k in shapes}
    (step/"params/_METADATA").write_text(json.dumps(dict(tree_metadata=tree)))
    (step/"params/array_metadatas/process_0").write_text(json.dumps(dict(array_metadatas=[
        dict(array_metadata=dict(param_name="params."+k+".value",write_shape=v)) for k,v in shapes.items()])))
    (step/"params/_sharding").write_text("fixture")
    (step/"params/manifest.ocdbt").write_text("fixture: no real weights")
    seal(root)


def seal(root):
    files = {str(p.relative_to(root)):package.sha256(p) for p in root.rglob("*") if p.is_file() and p.name!="MANIFEST.sha256.json"}
    (root/"MANIFEST.sha256.json").write_text(json.dumps(files))


def edit(root, file, values):
    path = root/file
    data = yaml.safe_load(path.read_text())
    data.update(values)
    path.write_text(yaml.safe_dump(data) if file.endswith("yaml") else json.dumps(data))
    seal(root)


class Profiles(unittest.TestCase):
    def test_shared_server_cli_uses_only_package_loader_and_warms_up(self):
        import sys
        path=ROOT/"docker/serve_checkpoint_package.py"
        module=importlib.util.module_from_spec(importlib.util.spec_from_file_location("package_server",path))
        module.__spec__.loader.exec_module(module)
        for robot in ("single","bimanual"):
            policy=mock.Mock(metadata=dict(robot=robot))
            policy.warm_up.return_value=1.0
            with self.subTest(robot=robot), \
                 mock.patch.object(module,"load_policy",return_value=policy) as load, \
                 mock.patch("openpi.serving.websocket_policy_server.WebsocketPolicyServer") as server, \
                 mock.patch.object(sys,"argv",["serve","--package","/package","--robot",robot]):
                module.main()
                load.assert_called_once_with(Path("/package"),expected_robot=robot)
                policy.warm_up.assert_called_once_with(None)
                server.return_value.serve_forever.assert_called_once()

    def test_profile_defaults_follow_vendored_model_definition(self):
        from openpi.models.pi0_config import Pi0Config
        for profile in package.MODEL_PROFILES.values():
            model = Pi0Config(pi05=profile["pi05"])
            for key in ("max_token_len", "discrete_state_input"):
                self.assertEqual(getattr(model,key),profile[key])

    def test_eight_profiles_config_transform_shape_and_action_order(self):
        from openpi import transforms
        from openpi.shared import normalize
        path = ROOT/"scripts/verify_checkpoint_package.py"
        module = importlib.util.module_from_spec(importlib.util.spec_from_file_location("verify_package", path))
        module.__spec__.loader.exec_module(module)
        for robot in ("single","bimanual"):
            for family in ("pi0","pi05"):
                for method in ("lora","full"):
                    with self.subTest(robot=robot,family=family,method=method), tempfile.TemporaryDirectory() as directory:
                        root = Path(directory)
                        fixture(root,robot,family,method,horizon=20 if robot=="single" else 50)
                        self.assertEqual(package.detect_format(root),"training_package")
                        spec = package.inspect_package(root)
                        self.assertEqual(spec["ranks"],[16,32] if method=="lora" else [None,None])
                        self.assertEqual(spec["contract"]["state_dim"],7 if robot=="single" else 14)
                        self.assertEqual(Path(spec["checkpoint"]),root/"checkpoint/39999")
                        config = package.build_train_config(spec,TOKENIZER)
                        self.assertEqual(config.model.pi05,family=="pi05")
                        self.assertEqual(config.model.action_horizon,20 if robot=="single" else 50)
                        self.assertEqual("lora" in config.model.paligemma_variant,method=="lora")
                        data = config.data.create(config.assets_dirs, config.model)
                        self.assertEqual(type(data.data_transforms.inputs[0]).__name__,"LiberoInputs" if robot=="single" else "AlohaInputs")
                        stats = normalize.load(Path(spec["stats_path"]).parent)
                        class FakePolicy:
                            metadata = {}
                            def infer(self, obs):
                                encoded = transforms.compose([*data.data_transforms.inputs,
                                    transforms.Normalize(stats,use_quantiles=data.use_quantile_norm),*data.model_transforms.inputs])(obs)
                                # Fake normalized output only, never random model weights.
                                result = dict(state=encoded["state"],actions=np.zeros((config.model.action_horizon,config.model.action_dim)))
                                return transforms.compose([transforms.Unnormalize(stats,use_quantiles=data.use_quantile_norm),*data.data_transforms.outputs])(result)
                        policy = package.PackagePolicy(FakePolicy(),spec,config,stats)
                        obs = module.observation(policy,"Move the object.")
                        raw = data.data_transforms.inputs[0](policy.adapt(obs))
                        np.testing.assert_array_equal(raw["state"],obs["observation/state"])
                        wire_keys = (["observation/front_image","observation/left_wrist_image","observation/right_wrist_image"]
                                     if robot=="bimanual" else ["observation/image","observation/wrist_image"])
                        for model_key,wire_key in zip(("base_0_rgb","left_wrist_0_rgb","right_wrist_0_rgb"),wire_keys):
                            np.testing.assert_array_equal(raw["image"][model_key],obs[wire_key])
                        output = policy.infer(obs)["actions"]
                        self.assertEqual(output.shape,(config.model.action_horizon,spec["contract"]["action_dim"]))
                        self.assertTrue(np.isfinite(output).all())
                        # Zero normalized output -> q midpoint or mean. The
                        # absolute grippers must never have State added again.
                        # Quantile inverse includes OpenPI's 1e-6 denominator
                        # stabilizer; allow its sub-micrometre offset.
                        np.testing.assert_allclose(output[:,6::7],.1,atol=2e-6)
                        self.assertGreaterEqual(policy.warm_up(),0)
                        result = module.transform_roundtrip(policy,obs)
                        self.assertTrue(result["passed"])
                        self.assertEqual(result["image_masks"]["right_wrist_0_rgb"],robot=="bimanual")

    def test_absolute_actions_and_nondefault_horizon_internal_dimension(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fixture(root,"single","pi0","full",horizon=13,delta=False,internal=16)
            spec=package.inspect_package(root)
            self.assertEqual(spec["delta_mask"],[])
            config=package.build_train_config(spec,TOKENIZER)
            self.assertEqual((config.model.action_horizon,config.model.action_dim),(13,16))
            data=config.data.create(config.assets_dirs,config.model)
            self.assertEqual([type(t).__name__ for t in data.data_transforms.outputs],["LiberoOutputs"])

    def test_fail_closed(self):
        cases=[("metadata.json",dict(robot="single")),("training.yaml",dict(method="full")),
               ("training.yaml",dict(lora_rank=8)),("training.yaml",dict(action_format="ee_delta")),
               ("training.yaml",dict(adapt_to_pi=True)),("training.yaml",dict(cameras=["front"])),
               ("tuning_norm_config.json",dict(action_horizon=20)),
               ("tuning_norm_config.json",dict(normalization="mixed")),
               ("training.yaml",dict(model_config=dict(action_dim=14)))]
        for file,values in cases:
            with self.subTest(values=values),tempfile.TemporaryDirectory() as d:
                root=Path(d);fixture(root);edit(root,file,values)
                with self.assertRaises((ValueError,KeyError)): package.inspect_package(root)

    def test_hash_missing_file_stats_mismatch_ambiguous_steps_and_path_escape(self):
        for mode in ("hash","missing","stats","steps","escape","extra"):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                root=Path(d);fixture(root)
                if mode=="hash": (root/"training.yaml").write_text("tamper")
                if mode=="missing": (root/"checkpoint/39999/_CHECKPOINT_METADATA").unlink();seal(root)
                if mode=="stats": (root/"norm_stats.json").write_text("{}");seal(root)
                if mode=="steps": (root/"checkpoint/40000").mkdir();seal(root)
                if mode=="extra": (root/"extra").write_text("unlisted")
                if mode=="escape": (root/"MANIFEST.sha256.json").write_text(json.dumps({"../escape":"0"*64}))
                with self.assertRaises((ValueError,FileNotFoundError)): package.inspect_package(root)

    def test_metadata_selects_step_and_legacy_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fixture(root);(root/"checkpoint/40000").mkdir()
            edit(root,"metadata.json",dict(checkpoint_step=39999))
            self.assertEqual(package.inspect_package(root)["step"],39999)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/"manifest.yaml").write_text(yaml.safe_dump(dict(contract={},checkpoint_path="checkpoint")))
            with self.assertRaisesRegex(ValueError,"no longer supported"):
                package.load_policy(root)

    def test_statistics_and_final_output_contract_are_rejected_when_invalid(self):
        for mode in ("width", "nonfinite", "bounds"):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                root=Path(d);fixture(root)
                path=root/"norm_stats.json"
                stats=json.loads(path.read_text())
                entry=stats["norm_stats"]["actions"]
                if mode=="width":entry["mean"]=[0.]*7
                if mode=="nonfinite":entry["q01"][0]=float("nan")
                if mode=="bounds":entry["q99"][0]=-10.
                path.write_text(json.dumps(stats))
                (root/"checkpoint/39999/assets/local/test/norm_stats.json").write_text(json.dumps(stats))
                seal(root)
                with self.assertRaises(ValueError):package.inspect_package(root)
        policy=package.PackagePolicy.__new__(package.PackagePolicy)
        policy.spec=dict(model=dict(action_horizon=50),contract=dict(action_dim=14))
        policy.adapt=lambda obs:obs
        for output in (np.zeros((50,32)),np.zeros((20,14)),np.full((50,14),np.nan)):
            policy.policy=mock.Mock(infer=lambda obs:dict(actions=output))
            with self.assertRaises(ValueError):policy.infer({})
        policy.policy=mock.Mock(infer=lambda obs:dict(actions=np.zeros((50,14))))
        self.assertEqual(policy.infer({})["actions"].shape,(50,14))

    def test_explicit_normalization_and_model_conflicts(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fixture(root,"single","pi0","full")
            edit(root,"training.yaml",dict(normalization="quantile",model_config=dict(max_token_len=96)))
            spec=package.inspect_package(root)
            self.assertEqual(spec["contract"]["normalization"],"quantile")
            self.assertEqual(spec["model"]["max_token_len"],96)
            edit(root,"metadata.json",dict(model_config=dict(max_token_len=48)))
            with self.assertRaisesRegex(ValueError,"conflicting model setting"):
                package.inspect_package(root)

    def test_launch_uses_readonly_package_no_build_and_external_runtime_files(self):
        for robot in ("single","bimanual"):
            with self.subTest(robot=robot),tempfile.TemporaryDirectory() as d:
                root=Path(d)/"package";root.mkdir();fixture(root,robot)
                token=Path(d)/"tokenizer";token.write_bytes(b"test tokenizer; no real model")
                output=Path(d)/"generated";output.mkdir()
                with mock.patch.dict(os.environ,{"PIPER_TOKENIZER_PATH":str(token)}), \
                     mock.patch.object(package,"TOKENIZER_SHA256",package.sha256(token)), \
                     mock.patch.object(package.tempfile,"mkdtemp",return_value=str(output)), \
                     mock.patch.object(package.subprocess,"run") as run:
                    package.launch(root,robot,ROOT)
                compose=yaml.safe_load((output/"compose.yaml").read_text())
                self.assertIn(str(root)+":/package:ro",compose["services"]["policy"]["volumes"])
                self.assertTrue((output/"manifest.yaml").is_file())
                self.assertFalse((root/"manifest.yaml").exists())
                config=yaml.safe_load((output/"runtime.yaml").read_text())
                self.assertTrue(config["runtime"]["dry_run"])
                calls=[c.args[0] for c in run.call_args_list]
                self.assertEqual(len(calls),3)
                self.assertIn("--no-build",calls[0])
                self.assertNotIn("--build",calls[0])
                self.assertIn("wait_policy_ready.py"," ".join(calls[1]))
                self.assertIn("client",calls[2])
                self.assertEqual(package.verify_manifest(root),json.loads((root/"MANIFEST.sha256.json").read_text()))

    def test_offline_launch_has_no_network_ros_or_client(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"package";root.mkdir();fixture(root)
            token=Path(d)/"tokenizer";token.write_bytes(b"fixture")
            output=Path(d)/"generated";output.mkdir()
            with mock.patch.dict(os.environ,{"PIPER_TOKENIZER_PATH":str(token)}), \
                 mock.patch.object(package,"TOKENIZER_SHA256",package.sha256(token)), \
                 mock.patch.object(package.tempfile,"mkdtemp",return_value=str(output)), \
                 mock.patch.object(package.subprocess,"run") as run:
                package.launch(root,"bimanual",ROOT,offline=True)
            args=run.call_args.args[0]
            self.assertEqual(args[args.index("--network")+1],"none")
            self.assertNotIn("--privileged",args)
            self.assertFalse(any("/dev/" in str(x) or "ROS_MASTER" in str(x) for x in args))
            self.assertFalse((output/"compose.yaml").exists())


if __name__ == "__main__":
    unittest.main()
