"""
Dual-arm real-robot rollout runner for Piper-like 14D checkpoints.

This script connects to the StarVLA websocket policy server and executes
returned actions on your real hardware adapter.

Expected action layout from piperx/arx_x5 checkpoints:
    [left_joints(6), left_gripper(1), right_joints(6), right_gripper(1)]
"""

from __future__ import annotations

import argparse
import ast
import importlib
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


class DualArmRobotAdapter(Protocol):
    """Robot interface this script expects."""

    def reset(self) -> None:
        """Move hardware to a safe reset pose."""

    def get_images(self) -> List[np.ndarray]:
        """Return [cam_high, cam_left_wrist, cam_right_wrist] RGB uint8 images."""

    def get_state(self) -> np.ndarray:
        """Return current 14D state matching training layout."""

    def step(self, action_14d: np.ndarray) -> None:
        """Execute one absolute 14D action on hardware."""

    def close(self) -> None:
        """Release resources (camera handles, robot drivers)."""


class ExamplePiperAdapter:
    """
    Placeholder adapter.

    Replace this with your AgileX Piper SDK implementation, or provide your own
    class via --adapter-module/--adapter-class.
    """

    def __init__(self, **kwargs: Any) -> None:
        _ = kwargs

    def reset(self) -> None:
        raise NotImplementedError("Implement reset() with your Piper SDK")

    def get_images(self) -> List[np.ndarray]:
        raise NotImplementedError("Implement get_images() with your camera stack")

    def get_state(self) -> np.ndarray:
        raise NotImplementedError("Implement get_state() returning 14D np.ndarray")

    def step(self, action_14d: np.ndarray) -> None:
        _ = action_14d
        raise NotImplementedError("Implement step(action_14d) with your Piper SDK")

    def close(self) -> None:
        return


@dataclass
class RolloutConfig:
    instruction: str
    episodes: int
    max_steps: int
    control_hz: float
    unnorm_key: str
    action_reorder: bool


def _load_adapter(module_name: str, class_name: str, adapter_kwargs: Dict[str, Any]) -> DualArmRobotAdapter:
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls(**adapter_kwargs)


def _parse_actions(response: Dict[str, Any]) -> np.ndarray:
    data = response.get("data", response)
    if "actions" not in data:
        raise KeyError(f"Response does not contain 'actions'. Keys: {list(data.keys())}")
    actions = np.asarray(data["actions"])
    if actions.ndim == 3:
        actions = actions[0]  # [B, T, D] -> [T, D]
    elif actions.ndim == 1:
        actions = actions.reshape(1, -1)  # [D] -> [1, D]
    if actions.shape[-1] != 14:
        raise ValueError(f"Expected action_dim=14, got {actions.shape}")
    return actions


def _maybe_reorder_action(action: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return action
    # Matches examples/Robotwin/eval_files/model2robotwin_interface.py ordering.
    reorder_idx = [0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]
    return action[reorder_idx]


def _build_request(images: List[np.ndarray], state: np.ndarray, instruction: str, unnorm_key: str) -> Dict[str, Any]:
    return {
        "examples": [
            {
                "image": images,
                "state": state,
                "lang": instruction,
            }
        ],
        "do_sample": False,
        "use_ddim": True,
        "num_ddim_steps": 10,
        "unnorm_key": unnorm_key,
    }


def run_rollout(client: WebsocketClientPolicy, robot: DualArmRobotAdapter, cfg: RolloutConfig) -> None:
    control_dt = 1.0 / cfg.control_hz if cfg.control_hz > 0 else 0.0

    for episode_idx in range(cfg.episodes):
        print(f"\n=== Episode {episode_idx + 1}/{cfg.episodes} ===")
        robot.reset()
        step_idx = 0
        while step_idx < cfg.max_steps:
            loop_start = time.time()

            images = robot.get_images()
            state = np.asarray(robot.get_state(), dtype=np.float32)
            if state.shape[-1] != 14:
                raise ValueError(f"Expected 14D state, got shape={state.shape}")

            request = _build_request(images=images, state=state, instruction=cfg.instruction, unnorm_key=cfg.unnorm_key)
            response = client.predict_action(request)
            action_chunk = _parse_actions(response)

            for action in action_chunk:
                mapped_action = _maybe_reorder_action(np.asarray(action, dtype=np.float32), cfg.action_reorder)
                robot.step(mapped_action)
                step_idx += 1
                if step_idx >= cfg.max_steps:
                    break

            elapsed = time.time() - loop_start
            if control_dt > elapsed:
                time.sleep(control_dt - elapsed)

        print(f"Episode complete, executed steps: {step_idx}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5694)
    parser.add_argument("--instruction", type=str, required=True)
    parser.add_argument("--unnorm_key", type=str, default="arx_x5")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--control_hz", type=float, default=10.0)
    parser.add_argument("--adapter-module", type=str, default="deployment.model_server.tools.rollout_piper_real")
    parser.add_argument("--adapter-class", type=str, default="ExamplePiperAdapter")
    parser.add_argument("--adapter-kwargs", type=str, default="{}", help="Python dict literal, e.g. '{\"ip\": \"192.168.1.10\"}'")
    parser.add_argument(
        "--action-reorder",
        action="store_true",
        help="Enable Robotwin index remap [0..5,12,6..11,13] if your controller needs it.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    adapter_kwargs = ast.literal_eval(args.adapter_kwargs)
    if not isinstance(adapter_kwargs, dict):
        raise ValueError("--adapter-kwargs must evaluate to a dict")

    cfg = RolloutConfig(
        instruction=args.instruction,
        episodes=args.episodes,
        max_steps=args.max_steps,
        control_hz=args.control_hz,
        unnorm_key=args.unnorm_key,
        action_reorder=args.action_reorder,
    )

    print(f"Connecting to policy server: ws://{args.host}:{args.port}")
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"Server metadata: {client.get_server_metadata()}")

    robot = _load_adapter(args.adapter_module, args.adapter_class, adapter_kwargs)
    try:
        run_rollout(client=client, robot=robot, cfg=cfg)
    finally:
        try:
            robot.close()
        finally:
            client.close()


if __name__ == "__main__":
    main()
