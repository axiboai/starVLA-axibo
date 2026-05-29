#!/usr/bin/env python3
"""
CosmoPredict2PI RGB policy inference for the bimanual Piper rig (starVLA WebSocket client).

Adapted from the openpi π₀.₅ depth inference script. Same ZMQ robot/camera stack,
but connects to ``deployment/model_server/server_policy.py`` instead of openpi.

Architecture
------------
    state JSON      tcp://localhost:3335
    cam_front       color tcp://localhost:5560
    cam_left_wrist  color tcp://localhost:5556
    cam_right_wrist color tcp://localhost:5558

    target PUB      tcp://0.0.0.0:3336  → follower_sink.py

    policy server   ws://localhost:5694  (starVLA CosmoPredict2PI)

Pre-flight
----------
- follower_sink.py on tcp://localhost:3336
- RealSense **color** publishers on all three cameras (depth not used)
- Teleop publisher on 3335 (follower state; use ``--no-command`` on 3336)
- starVLA policy server:

    cd /workspace/starVLA-axibo
    export PYTHONPATH=.
    python deployment/model_server/server_policy.py \\
        --ckpt_path playground/Checkpoints/.../checkpoints/steps_80000_pytorch_model.pt \\
        --port 5694 --use_bf16

Run
---
    cd /workspace/starVLA-axibo
    export PYTHONPATH=.
    python deployment/model_server/tools/cosmos_pi_piper_zmq_inference.py \\
        --instruction "fold the towel"
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys

from deployment.model_server.tools import rollout_piper_real
from deployment.model_server.tools.piper_zmq_adapter import PiperZmqAdapter
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--host", default="127.0.0.1", help="starVLA policy server host")
    p.add_argument("--port", type=int, default=5694, help="starVLA policy server port")
    p.add_argument("--instruction", default="fold the towel")
    p.add_argument("--unnorm_key", default="arx_x5")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--control_hz", type=float, default=5.0)
    p.add_argument("--chunk-execute-steps", type=int, default=4)
    p.add_argument("--max-joint-delta", type=float, default=0.05)
    p.add_argument("--smooth-alpha", type=float, default=0.35)
    p.add_argument("--no-gripper-snap", action="store_true")
    p.add_argument("--max-chunks", type=int, default=None)
    p.add_argument(
        "--wrist-only",
        action="store_true",
        help="Only wrist RGB publishers; duplicate a wrist feed into cam_high.",
    )
    p.add_argument(
        "--front-from",
        choices=("left", "right", "none"),
        default="left",
        help="With --wrist-only, which wrist fills cam_high (default: left).",
    )
    p.add_argument("--state-addr", default="tcp://localhost:3335")
    p.add_argument("--target-addr", default="tcp://0.0.0.0:3336")
    p.add_argument("--cam_front_color_addr", default="tcp://localhost:5560")
    p.add_argument("--cam_left_wrist_color_addr", default="tcp://localhost:5556")
    p.add_argument("--cam_right_wrist_color_addr", default="tcp://localhost:5558")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print(f"[infer] Connecting to starVLA policy server at {args.host}:{args.port} ...", flush=True)
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    server_meta = client.get_server_metadata()
    print(f"[infer] Server metadata: {server_meta}", flush=True)

    action_chunk_size = int(server_meta.get("action_chunk_size", 16))
    chunk_execute_steps = args.chunk_execute_steps if args.chunk_execute_steps > 0 else action_chunk_size

    cfg = rollout_piper_real.RolloutConfig(
        instruction=args.instruction,
        episodes=args.episodes,
        max_steps=args.max_steps,
        max_chunks=args.max_chunks,
        control_hz=args.control_hz,
        unnorm_key=args.unnorm_key,
        action_reorder=False,
        include_state=False,
        action_chunk_size=action_chunk_size,
        chunk_execute_steps=chunk_execute_steps,
        max_joint_delta=args.max_joint_delta,
        smooth_alpha=args.smooth_alpha,
        gripper_snap=not args.no_gripper_snap,
    )

    robot = PiperZmqAdapter(
        state_addr=args.state_addr,
        target_addr=args.target_addr,
        cam_front_color_addr=args.cam_front_color_addr,
        cam_left_wrist_color_addr=args.cam_left_wrist_color_addr,
        cam_right_wrist_color_addr=args.cam_right_wrist_color_addr,
        wrist_only=args.wrist_only,
        front_from=args.front_from,
    )

    stop = {"flag": False}

    def _stop(*_):
        print("[infer] stop requested", flush=True)
        stop["flag"] = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    print(
        f"[infer] CosmoPredict2PI rollout  prompt={args.instruction!r}  "
        f"chunk_execute_steps={chunk_execute_steps}  control_hz={args.control_hz}",
        flush=True,
    )

    try:
        rollout_piper_real.run_rollout(client=client, robot=robot, cfg=cfg)
    except KeyboardInterrupt:
        pass
    finally:
        robot.close()
        client.close()
        print("[infer] done.", flush=True)


if __name__ == "__main__":
    main()
