"""Shell entry points forward to one package CLI; no Docker or model runs."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class LaunchArgumentsTests(unittest.TestCase):
    def test_both_scripts_forward_package_flags_to_shared_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            binary = temp / "python3"
            binary.write_text("#!/usr/bin/env python3\n")
            # Use the real interpreter in the shebang to avoid PATH recursion.
            import sys
            binary.write_text("#!" + sys.executable + "\nimport json,os,sys\n"
                              "with open(os.environ['CLI_TEST_LOG'],'w') as f: json.dump(sys.argv[1:],f)\n")
            binary.chmod(0o755)
            for script,robot in (("run.sh","single"),("run_dual.sh","bimanual")):
                log=temp/(robot+".json")
                env=dict(os.environ,PATH=str(temp)+":"+os.environ['PATH'],CLI_TEST_LOG=str(log))
                result=subprocess.run(["bash",str(ROOT/"scripts"/script),"--checkpoint",str(temp/"model"),
                    "--offline","--dry-run","--warmup-prompt","Move the object."],env=env,text=True,capture_output=True)
                self.assertEqual(result.returncode,0,result.stderr)
                args=json.loads(log.read_text())
                self.assertEqual(args[:2],["-m","piper_vla_common.checkpoint_package"])
                self.assertEqual(args[args.index("--robot")+1],robot)
                self.assertEqual(args[args.index("--checkpoint")+1],str(temp/"model"))
                self.assertIn("--offline",args)
                self.assertIn("--dry-run",args)
                self.assertNotIn("--build",args)
