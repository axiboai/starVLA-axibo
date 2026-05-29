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
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

logger = logging.getLogger(__name__)


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


class DryRunPiperAdapter:
    """No-hardware adapter for testing policy-server latency and action parsing."""

    def __init__(self, **kwargs: Any) -> None:
        _ = kwargs

    def reset(self) -> None:
        logger.info("dry-run reset")

    def get_images(self) -> List[np.ndarray]:
        blank = np.zeros((224, 224, 3), dtype=np.uint8)
        return [blank, blank, blank]

    def get_state(self) -> np.ndarray:
        return np.zeros((14,), dtype=np.float32)

    def step(self, action_14d: np.ndarray) -> None:
        logger.info("dry-run step action[:3]=%s gripper=%.3f", action_14d[:3], action_14d[6])

    def close(self) -> None:
        return


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
    max_chunks: Optional[int]
    control_hz: float
    unnorm_key: str
    action_reorder: bool
    include_state: bool
    action_chunk_size: int
    chunk_execute_steps: Optional[int]
    max_joint_delta: float
    smooth_alpha: float
    gripper_snap: bool


def _load_adapter(module_name: str, class_name: str, adapter_kwargs: Dict[str, Any]) -> DualArmRobotAdapter:
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls(**adapter_kwargs)


def _parse_actions(response: Dict[str, Any]) -> np.ndarray:
    if response.get("status") == "error":
        err = response.get("error", response)
        raise RuntimeError(f"Policy server error: {err}")

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
    # Robotwin AgileX sim only — do NOT use for real Piper/arx_x5 joint layout.
    reorder_idx = [0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]
    return action[reorder_idx]


def _snap_gripper(value: float, prev: Optional[float]) -> float:
    """Reduce gripper chatter with simple hysteresis."""
    open_th = 0.55
    close_th = 0.45
    if prev is None:
        return 1.0 if value >= 0.5 else 0.0
    if prev >= 0.5:
        return 1.0 if value >= close_th else 0.0
    return 1.0 if value >= open_th else 0.0


def _postprocess_action(
    target: np.ndarray,
    current_state: np.ndarray,
    prev_command: Optional[np.ndarray],
    cfg: RolloutConfig,
    prev_grippers: tuple[Optional[float], Optional[float]],
) -> tuple[np.ndarray, tuple[Optional[float], Optional[float]]]:
    """Clip/smooth absolute joint targets against live proprio to avoid jitter."""
    cmd = np.asarray(target, dtype=np.float32).reshape(-1).copy()
    qpos = np.asarray(current_state, dtype=np.float32).reshape(-1).copy()
    if cmd.shape != qpos.shape:
        raise ValueError(f"Action/state dim mismatch: action={cmd.shape}, state={qpos.shape}")

    left_joints = slice(0, 6)
    right_joints = slice(7, 13)

    if cfg.max_joint_delta > 0:
        for joint_slice in (left_joints, right_joints):
            delta = cmd[joint_slice] - qpos[joint_slice]
            delta = np.clip(delta, -cfg.max_joint_delta, cfg.max_joint_delta)
            cmd[joint_slice] = qpos[joint_slice] + delta

    if cfg.smooth_alpha > 0 and prev_command is not None:
        prev_command = np.asarray(prev_command, dtype=np.float32).reshape(-1)
        blend = cfg.smooth_alpha
        cmd[left_joints] = blend * cmd[left_joints] + (1.0 - blend) * prev_command[left_joints]
        cmd[right_joints] = blend * cmd[right_joints] + (1.0 - blend) * prev_command[right_joints]

    if cfg.gripper_snap:
        lg_prev, rg_prev = prev_grippers
        cmd[6] = _snap_gripper(float(cmd[6]), lg_prev)
        cmd[13] = _snap_gripper(float(cmd[13]), rg_prev)
        prev_grippers = (float(cmd[6]), float(cmd[13]))
    else:
        prev_grippers = (float(cmd[6]), float(cmd[13]))

    max_delta = float(np.max(np.abs(cmd[[*range(6), *range(7, 13)]] - qpos[[*range(6), *range(7, 13)]])))
    logger.debug("postprocess max joint delta vs proprio: %.4f rad", max_delta)
    return cmd, prev_grippers


def _build_request(
    images: List[np.ndarray],
    instruction: str,
    unnorm_key: str,
    state: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    example: Dict[str, Any] = {"image": images, "lang": instruction}
    if state is not None:
        example["state"] = state

    return {
        "examples": [example],
        "do_sample": False,
        "unnorm_key": unnorm_key,
    }


def _request_inference(
    client: WebsocketClientPolicy,
    robot: DualArmRobotAdapter,
    cfg: RolloutConfig,
) -> np.ndarray:
    obs_start = time.time()
    images = robot.get_images()
    if len(images) != 3:
        raise ValueError(f"Expected 3 camera views, got {len(images)}")

    example_state: Optional[np.ndarray] = None
    if cfg.include_state:
        state = np.asarray(robot.get_state(), dtype=np.float32).reshape(-1)
        if state.shape[-1] != 14:
            raise ValueError(f"Expected 14D state, got shape={state.shape}")
        example_state = state

    obs_elapsed = time.time() - obs_start
    logger.info("observation ready in %.2fs (include_state=%s)", obs_elapsed, cfg.include_state)

    request = _build_request(
        images=images,
        instruction=cfg.instruction,
        unnorm_key=cfg.unnorm_key,
        state=example_state,
    )

    logger.info("waiting for CosmoPredict2PI inference (this can take 30-120s on A100)...")
    infer_start = time.time()
    response = client.predict_action(request)
    infer_elapsed = time.time() - infer_start
    logger.info("inference finished in %.2fs", infer_elapsed)

    return _parse_actions(response)


def run_rollout(client: WebsocketClientPolicy, robot: DualArmRobotAdapter, cfg: RolloutConfig) -> None:
    control_dt = 1.0 / cfg.control_hz if cfg.control_hz > 0 else 0.0
    cached_chunk: Optional[np.ndarray] = None
    chunk_idx = 0
    chunks_run = 0
    prev_command: Optional[np.ndarray] = None
    prev_grippers: tuple[Optional[float], Optional[float]] = (None, None)
    chunk_execute_steps = cfg.chunk_execute_steps or cfg.action_chunk_size

    for episode_idx in range(cfg.episodes):
        logger.info("=== Episode %d/%d ===", episode_idx + 1, cfg.episodes)
        robot.reset()
        step_idx = 0
        cached_chunk = None
        chunk_idx = 0
        chunks_run = 0
        prev_command = None
        prev_grippers = (None, None)

        while step_idx < cfg.max_steps:
            loop_start = time.time()

            if cached_chunk is None or chunk_idx >= min(len(cached_chunk), chunk_execute_steps):
                if cfg.max_chunks is not None and chunks_run >= cfg.max_chunks:
                    logger.info("Reached max_chunks=%d, stopping episode.", cfg.max_chunks)
                    break

                cached_chunk = _request_inference(client, robot, cfg)
                chunk_idx = 0
                chunks_run += 1
                logger.info(
                    "chunk %d received: shape=%s (executing first %d/%d steps before replan)",
                    chunks_run,
                    cached_chunk.shape,
                    min(chunk_execute_steps, len(cached_chunk)),
                    len(cached_chunk),
                )

            action = cached_chunk[chunk_idx]
            chunk_idx += 1

            exec_start = time.time()
            mapped_action = _maybe_reorder_action(np.asarray(action, dtype=np.float32), cfg.action_reorder)
            current_state = np.asarray(robot.get_state(), dtype=np.float32).reshape(-1)
            safe_action, prev_grippers = _postprocess_action(
                target=mapped_action,
                current_state=current_state,
                prev_command=prev_command,
                cfg=cfg,
                prev_grippers=prev_grippers,
            )
            robot.step(safe_action)
            prev_command = safe_action.copy()
            exec_elapsed = time.time() - exec_start
            step_idx += 1

            logger.info(
                "step %d/%d (chunk %d, idx %d) cmd-vs-proprio max=%.3f robot.step=%.2fs",
                step_idx,
                cfg.max_steps,
                chunks_run,
                chunk_idx - 1,
                float(np.max(np.abs(safe_action[[*range(6), *range(7, 13)]] - current_state[[*range(6), *range(7, 13)]]))),
                exec_elapsed,
            )

            elapsed = time.time() - loop_start
            if control_dt > elapsed:
                time.sleep(control_dt - elapsed)

        logger.info("Episode complete, executed steps: %d, inference calls: %d", step_idx, chunks_run)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5694)
    parser.add_argument("--instruction", type=str, required=True)
    parser.add_argument("--unnorm_key", type=str, default="arx_x5")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--control_hz", type=float, default=5.0, help="Command rate. Lower is safer on real arms.")
    parser.add_argument(
        "--chunk-execute-steps",
        type=int,
        default=4,
        help="How many actions from each predicted chunk to execute before replanning with fresh cameras.",
    )
    parser.add_argument(
        "--max-joint-delta",
        type=float,
        default=0.05,
        help="Max per-step joint move (rad) relative to current proprio. Set 0 to disable clipping.",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.35,
        help="EMA blend for joint targets (0=off, 0.35=moderate smoothing).",
    )
    parser.add_argument(
        "--no-gripper-snap",
        action="store_true",
        help="Disable gripper hysteresis snap (open/close thresholding).",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Stop after this many policy inferences (useful for smoke tests).",
    )
    parser.add_argument(
        "--include-state",
        action="store_true",
        help="Send proprio state to server. Piper cosmo training used include_state=false, so leave this off by default.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use DryRunPiperAdapter (no robot motion) to test server latency.",
    )
    parser.add_argument("--adapter-module", type=str, default="deployment.model_server.tools.rollout_piper_real")
    parser.add_argument("--adapter-class", type=str, default="ExamplePiperAdapter")
    parser.add_argument("--adapter-kwargs", type=str, default="{}", help="Python dict literal, e.g. '{\"ip\": \"192.168.1.10\"}'")
    parser.add_argument(
        "--action-reorder",
        action="store_true",
        help="Robotwin AgileX sim index remap only. Leave OFF for real Piper/arx_x5.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    adapter_kwargs = ast.literal_eval(args.adapter_kwargs)
    if not isinstance(adapter_kwargs, dict):
        raise ValueError("--adapter-kwargs must evaluate to a dict")

    logger.info("Connecting to policy server: ws://%s:%d", args.host, args.port)
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    server_meta = client.get_server_metadata()
    logger.info("Server metadata: %s", server_meta)

    action_chunk_size = int(server_meta.get("action_chunk_size", 16))
    chunk_execute_steps = args.chunk_execute_steps if args.chunk_execute_steps > 0 else action_chunk_size
    logger.info(
        "Real-robot tips: CosmoPredict2PI is slow (~30-120s/chunk). "
        "Use small --chunk-execute-steps (default 4) and --max-joint-delta to avoid jitter."
    )

    cfg = RolloutConfig(
        instruction=args.instruction,
        episodes=args.episodes,
        max_steps=args.max_steps,
        max_chunks=args.max_chunks,
        control_hz=args.control_hz,
        unnorm_key=args.unnorm_key,
        action_reorder=args.action_reorder,
        include_state=args.include_state,
        action_chunk_size=action_chunk_size,
        chunk_execute_steps=chunk_execute_steps,
        max_joint_delta=args.max_joint_delta,
        smooth_alpha=args.smooth_alpha,
        gripper_snap=not args.no_gripper_snap,
    )

    if args.dry_run:
        robot: DualArmRobotAdapter = DryRunPiperAdapter(**adapter_kwargs)
    else:
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
