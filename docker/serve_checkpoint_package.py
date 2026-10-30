#!/usr/bin/env python3
"""Serve single/dual pi0/pi05 Full/LoRA training packages through OpenPI."""
import argparse
from pathlib import Path

from piper_vla_common.checkpoint_package import load_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--robot", required=True, choices=("single", "bimanual"))
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--warmup-prompt")
    args = parser.parse_args()
    policy = load_policy(args.package, expected_robot=args.robot)
    warmup_ms = policy.warm_up(args.warmup_prompt)
    print("Policy ready: warm-up %.1f ms; metadata=%s" % (warmup_ms, policy.metadata), flush=True)
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    WebsocketPolicyServer(policy, host="0.0.0.0", port=args.port,
                          metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    main()
