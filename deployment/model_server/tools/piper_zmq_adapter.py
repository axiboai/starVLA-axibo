"""
ZMQ Piper hardware adapter for starVLA CosmoPredict2PI rollout.

Reuses the bimanual Piper ZMQ stack (follower state + RealSense color streams)
but **RGB only** — no depth. Implements the interface expected by
``rollout_piper_real.py``.

ZMQ layout (same as openpi depth inference, minus depth ports)
--------------------------------------------------------------
    state JSON      tcp://localhost:3335
    cam_front       color tcp://localhost:5560
    cam_left_wrist  color tcp://localhost:5556
    cam_right_wrist color tcp://localhost:5558

    target PUB      tcp://0.0.0.0:3336  → follower_sink.py

Training camera order (cam_high, cam_left_wrist, cam_right_wrist):
    cam_front → cam_high, then left wrist, then right wrist.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import zmq

logger = logging.getLogger(__name__)

TELEOP_STATE_ADDR = "tcp://localhost:3335"
TARGET_PUB_ADDR = "tcp://0.0.0.0:3336"

CAMERA_COLOR_PORTS: dict[str, str] = {
    "cam_front": "tcp://localhost:5560",
    "cam_left_wrist": "tcp://localhost:5556",
    "cam_right_wrist": "tcp://localhost:5558",
}
# Returned to starVLA in training order: high, left wrist, right wrist.
CAMERA_ORDER = ("cam_front", "cam_left_wrist", "cam_right_wrist")

RENDER_HEIGHT = 224
RENDER_WIDTH = 224


class ConflateSub:
    """SUB socket in conflate mode (latest teleop JSON only)."""

    def __init__(self, ctx: zmq.Context, addr: str):
        s = ctx.socket(zmq.SUB)
        s.setsockopt(zmq.CONFLATE, 1)
        s.setsockopt(zmq.RCVHWM, 1)
        s.setsockopt(zmq.LINGER, 0)
        s.connect(addr)
        s.setsockopt(zmq.SUBSCRIBE, b"")
        self.s = s
        self._last_text: Optional[str] = None

    def poll(self) -> Optional[str]:
        latest = None
        while True:
            try:
                latest = self.s.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
        if latest is not None:
            self._last_text = latest
        return latest

    def latest(self) -> Optional[str]:
        self.poll()
        return self._last_text

    def close(self) -> None:
        try:
            self.s.close(linger=0)
        except Exception:
            pass


class CameraSub:
    """SUB for one RealSense color stream (topic ``image``)."""

    def __init__(self, ctx: zmq.Context, addr: str, rgb_topic: bytes = b"image"):
        s = ctx.socket(zmq.SUB)
        s.setsockopt(zmq.RCVHWM, 1)
        s.setsockopt(zmq.LINGER, 0)
        s.connect(addr)
        s.setsockopt(zmq.SUBSCRIBE, rgb_topic)
        self.s = s
        self.rgb_topic = rgb_topic
        self.last_rgb: Optional[np.ndarray] = None

    def poll(self) -> bool:
        got_new = False
        while True:
            try:
                parts = self.s.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            if len(parts) < 3 or parts[0] != self.rgb_topic or len(parts[1]) != 8:
                continue
            payload = parts[2]
            if len(payload) < 2 or payload[:2] != b"\xff\xd8":
                continue
            arr = np.frombuffer(payload, dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            self.last_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            got_new = True
        return got_new

    def ok(self) -> bool:
        return self.last_rgb is not None

    def close(self) -> None:
        try:
            self.s.close(linger=0)
        except Exception:
            pass


def pack_state_14(state_msg: dict) -> np.ndarray:
    fL = state_msg["follower_left"]
    fR = state_msg["follower_right"]
    out = np.zeros(14, dtype=np.float32)
    out[:6] = fL["q"][:6]
    out[6] = fL["gripper"]
    out[7:13] = fR["q"][:6]
    out[13] = fR["gripper"]
    return out


def resize_rgb_hwc(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    if rgb.shape[0] == height and rgb.shape[1] == width:
        return rgb
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)


def resolve_cam_front_rgb(
    cam_front_rgb: Optional[np.ndarray],
    cam_left_rgb: Optional[np.ndarray],
    cam_right_rgb: Optional[np.ndarray],
    *,
    wrist_only: bool = False,
    front_from: str = "left",
) -> Optional[np.ndarray]:
    if not wrist_only:
        return cam_front_rgb
    if front_from == "none":
        return None
    if front_from == "right":
        return cam_right_rgb
    return cam_left_rgb


def make_target_msg(seq: int, action_14: np.ndarray) -> str:
    action_14 = np.asarray(action_14, dtype=np.float32)
    return json.dumps(
        {
            "t_mono": time.monotonic(),
            "seq": seq,
            "source": "policy",
            "left": {"q": action_14[0:6].tolist(), "gripper": float(action_14[6])},
            "right": {"q": action_14[7:13].tolist(), "gripper": float(action_14[13])},
        }
    )


class PiperZmqAdapter:
    """
    Hardware adapter for ``rollout_piper_real.py``.

    kwargs (via ``--adapter-kwargs``):
        state_addr, target_addr, cam_*_color_addr, wrist_only, front_from,
        image_height, image_width, warmup_timeout_s
    """

    def __init__(
        self,
        state_addr: str = TELEOP_STATE_ADDR,
        target_addr: str = TARGET_PUB_ADDR,
        cam_front_color_addr: str = CAMERA_COLOR_PORTS["cam_front"],
        cam_left_wrist_color_addr: str = CAMERA_COLOR_PORTS["cam_left_wrist"],
        cam_right_wrist_color_addr: str = CAMERA_COLOR_PORTS["cam_right_wrist"],
        wrist_only: bool = False,
        front_from: str = "left",
        image_height: int = RENDER_HEIGHT,
        image_width: int = RENDER_WIDTH,
        warmup_timeout_s: float = 5.0,
        **kwargs: Any,
    ) -> None:
        _ = kwargs
        self.wrist_only = wrist_only
        self.front_from = front_from
        self.image_height = image_height
        self.image_width = image_width
        self.warmup_timeout_s = warmup_timeout_s

        self._ctx = zmq.Context.instance()
        self._sub_state = ConflateSub(self._ctx, state_addr)

        color_addrs = {
            "cam_front": cam_front_color_addr,
            "cam_left_wrist": cam_left_wrist_color_addr,
            "cam_right_wrist": cam_right_wrist_color_addr,
        }
        self._cams: Dict[str, CameraSub] = {}
        for cam in CAMERA_ORDER:
            if wrist_only and cam == "cam_front":
                continue
            self._cams[cam] = CameraSub(self._ctx, color_addrs[cam])

        self._pub_target = self._ctx.socket(zmq.PUB)
        self._pub_target.setsockopt(zmq.SNDHWM, 1)
        self._pub_target.setsockopt(zmq.LINGER, 0)
        self._pub_target.bind(target_addr)

        self._state_dict: Optional[dict] = None
        self._seq = 0

        if wrist_only:
            src = "black" if front_from == "none" else f"{front_from} wrist"
            logger.info("wrist-only mode: cam_high (cam_front) from %s", src)
        logger.info("target PUB bound to %s", target_addr)

        self._warmup()

    def _poll_inputs(self) -> None:
        state_text = self._sub_state.latest()
        if state_text is not None:
            try:
                self._state_dict = json.loads(state_text)
            except json.JSONDecodeError:
                pass
        for cam in self._cams.values():
            cam.poll()

    def _cameras_ok(self) -> bool:
        if self.wrist_only:
            return (
                self._cams["cam_left_wrist"].ok()
                and self._cams["cam_right_wrist"].ok()
            )
        return all(c.ok() for c in self._cams.values())

    def _warmup(self) -> None:
        logger.info("warming up ZMQ state + RGB cameras...")
        deadline = time.monotonic() + self.warmup_timeout_s
        while time.monotonic() < deadline:
            self._poll_inputs()
            if self._state_dict is not None and self._cameras_ok():
                logger.info("ZMQ warmup complete.")
                return
            time.sleep(0.05)
        logger.warning("ZMQ warmup timed out — state or cameras may be missing.")

    def reset(self) -> None:
        self._poll_inputs()
        if self._state_dict is not None:
            hold = pack_state_14(self._state_dict)
            self.step(hold)
            logger.info("reset: holding current follower pose")

    def get_images(self) -> List[np.ndarray]:
        self._poll_inputs()

        left_rgb = self._cams["cam_left_wrist"].last_rgb
        right_rgb = self._cams["cam_right_wrist"].last_rgb
        front_rgb = resolve_cam_front_rgb(
            self._cams["cam_front"].last_rgb if "cam_front" in self._cams else None,
            left_rgb,
            right_rgb,
            wrist_only=self.wrist_only,
            front_from=self.front_from,
        )

        placeholder = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)

        def _prep(rgb: Optional[np.ndarray]) -> np.ndarray:
            if rgb is None:
                return placeholder
            return resize_rgb_hwc(rgb, self.image_height, self.image_width)

        # Training order: cam_high, cam_left_wrist, cam_right_wrist
        return [_prep(front_rgb), _prep(left_rgb), _prep(right_rgb)]

    def get_state(self) -> np.ndarray:
        self._poll_inputs()
        if self._state_dict is None:
            raise RuntimeError("No follower state on ZMQ yet (tcp://localhost:3335)")
        return pack_state_14(self._state_dict)

    def step(self, action_14d: np.ndarray) -> None:
        msg = make_target_msg(self._seq, action_14d)
        try:
            self._pub_target.send_string(msg, flags=zmq.NOBLOCK)
        except zmq.Again:
            logger.warning("target PUB HWM full, dropping seq=%d", self._seq)
        self._seq += 1

    def close(self) -> None:
        try:
            self._pub_target.close(linger=0)
        except Exception:
            pass
        self._sub_state.close()
        for cam in self._cams.values():
            cam.close()
