"""Static package-only distribution checks; never invoke Docker or models."""
import ast
from pathlib import Path
import shlex
import unittest

ROOT = Path(__file__).resolve().parents[2]


class DistributionTests(unittest.TestCase):
    def test_every_docker_copy_source_is_distributed(self):
        dockerfile=(ROOT/"docker/Dockerfile").read_text()
        for line in dockerfile.splitlines():
            if line.startswith("COPY ") and "--from=" not in line:
                for source in shlex.split(line)[1:-1]:
                    self.assertTrue((ROOT/source).exists(),source)
        for target in ("policy-base","dual-policy","ros-client-base","dual-client"):
            self.assertIn(" AS "+target+"\n",dockerfile)
        self.assertIn("serve_checkpoint_package.py",dockerfile)
        self.assertNotIn("training_registration",dockerfile)
        self.assertNotIn("piper_pi0_real_pickup",dockerfile)

    def test_no_production_import_of_removed_bundle_modules(self):
        removed={"bundle_policy","registration_compat","pi0_registration"}
        for folder in ("src","docker","scripts"):
            for path in (ROOT/folder).rglob("*.py"):
                for node in ast.walk(ast.parse(path.read_text())):
                    if isinstance(node,ast.Import):names=[a.name for a in node.names]
                    elif isinstance(node,ast.ImportFrom):names=[node.module or ""]
                    else:continue
                    for name in names:
                        self.assertFalse(set(name.split('.')) & removed,(path,name))
        source=(ROOT/"src/piper_vla_common/checkpoint_package.py").read_text()
        self.assertIn("serve_checkpoint_package.py",source)
        self.assertIn('"--no-build"',source)
        self.assertIn('"--pull", "never"',source)

    def test_scripts_keep_no_legacy_bundle_launch_paths(self):
        for name in ("run.sh","run_dual.sh"):
            text=(ROOT/"scripts"/name).read_text()
            self.assertIn("piper_vla_common.checkpoint_package",text)
            self.assertNotIn("manifest.yaml",text)
            self.assertNotIn("docker compose",text)
