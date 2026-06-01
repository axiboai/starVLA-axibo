#!/usr/bin/env python3
"""
Offline action-chunk probe for the CosmoPredict2PI policy server.

Purpose
-------
Bisect "arms idle / hold pose" failures into one of two buckets:

  (A) The MODEL is fine but the LIVE observation is out-of-distribution
      (wrong camera content/order, black feed, color/exposure mismatch).
  (B) The MODEL/checkpoint itself outputs a near-static chunk even on a
      KNOWN-GOOD training frame (data / convergence / embodiment-frame issue).

It does NOT touch the robot. It feeds a fixed set of 3 images + instruction
to the already-running policy server (same request path as the real rollout)
and reports how much MOTION is in each predicted chunk.

Read the numbers like this
--------------------------
  first_to_last : max per-joint displacement from chunk step 0 -> 15.
                  This is "how far does the policy intend to move this chunk".
  step_to_step  : mean per-step max-joint delta (chunk smoothness / speed).
  total_path    : summed per-step motion over the whole chunk.
  gripper_span  : max-min of each gripper over the chunk (grasp intent).

  Run it on a TRAINING frame (in-distribution):
    * first_to_last is sizeable (e.g. >> step_to_step, clear net direction)
        -> model is healthy; your LIVE pipeline is the problem  => case (A)
    * first_to_last ~= a few hundredths of a rad, no net direction,
      and roughly the SAME tiny number you saw live
        -> the checkpoint outputs "hold" even in-distribution => case (B)

We sample the chunk several times (flow-matching seeds fresh noise each call)
so you can also see how much of the motion is real signal vs. sampling noise:
if the mean first_to_last is tiny but the std is comparable, the chunk is
basically noise around a constant.

Usage
-----
    export PYTHONPATH=.
    # point at 3 images that match training camera order:
    python deployment/model_server/tools/offline_chunk_probe.py \
        --high  /path/cam_high.png \
        --left  /path/cam_left_wrist.png \
        --right /path/cam_right_wrist.png \
        --instruction "fold towel" \
        --samples 5

Tip: grab a real training frame to test case (B). The fastest way is to
decode one frame from each of the three episode videos in your
playground/Datasets/arx-x5 LeRobot dataset (e.g. with `--from-video`,
see below) so the input is guaranteed in-distribution.
"""

from __future__ import annotations

import argparse
from typing import List, Optional

import numpy as np
from PIL import Image

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

# Model trains actions GROUPED: [Ljoints(6), Rjoints(6), Lgrip, Rgrip].
# Joints are indices 0..11; grippers are 12 and 13 in that layout.
JOINT_IDX = list(range(12))
GRIPPER_IDX = [12, 13]


def load_image(path: str, size: int = 224) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def frame_from_video(path: str, frame_idx: int, size: int = 224) -> np.ndarray:
    """Decode a single frame from a video file (for pulling a training frame)."""
    import cv2

    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx} from {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.asarray(Image.fromarray(rgb).resize((size, size), Image.BILINEAR), dtype=np.uint8)


def parse_actions(response: dict) -> np.ndarray:
    if response.get("status") == "error":
        raise RuntimeError(f"Policy server error: {response.get('error', response)}")
    data = response.get("data", response)
    if "actions" not in data:
        raise KeyError(f"No 'actions' in response. Keys: {list(data.keys())}")
    actions = np.asarray(data["actions"], dtype=np.float64)
    if actions.ndim == 3:
        actions = actions[0]  # [B,T,D] -> [T,D]
    elif actions.ndim == 1:
        actions = actions.reshape(1, -1)
    return actions


def chunk_metrics(chunk: np.ndarray) -> dict:
    j = chunk[:, JOINT_IDX]
    per_step = np.max(np.abs(np.diff(j, axis=0)), axis=1)  # (T-1,)
    first_to_last = float(np.max(np.abs(j[-1] - j[0])))
    total_path = float(np.sum(per_step))
    # directionality: net displacement / summed path. UNIT-FREE, so it is the
    # one metric we can compare between a NORMALIZED gt chunk and the RAW rad
    # prediction. ~1 => clean reach (monotonic). ~0 => jitter/oscillation in place.
    directionality = first_to_last / total_path if total_path > 1e-9 else 0.0
    return {
        "first_to_last": first_to_last,
        "step_to_step": float(np.mean(per_step)),
        "total_path": total_path,
        "directionality": directionality,
        "grip_span": [float(chunk[:, g].max() - chunk[:, g].min()) for g in GRIPPER_IDX],
        "argmax_joint": int(np.argmax(np.abs(j[-1] - j[0]))),
    }


def report_gt_chunk(gt_chunk: np.ndarray) -> None:
    """Print metrics for the GROUND-TRUTH action chunk pulled from the dataset.

    gt_chunk must be [T, 14] in the GROUPED training layout
    [Ljoints(6), Rjoints(6), Lgrip, Rgrip] -- the same layout chunk_metrics uses.
    The GT is usually min_max-NORMALIZED to [-1,1]; that's fine: we compare it to
    the prediction via `directionality`, which is scale-free.
    """
    gt_chunk = np.asarray(gt_chunk, dtype=np.float64)
    if gt_chunk.ndim == 3:
        gt_chunk = gt_chunk[0]
    m = chunk_metrics(gt_chunk)
    looks_norm = float(np.max(np.abs(gt_chunk[:, JOINT_IDX]))) <= 1.5
    print("-" * 78)
    print(
        f"GROUND TRUTH chunk: shape={gt_chunk.shape}  "
        f"({'normalized [-1,1]' if looks_norm else 'raw units'})"
    )
    print(
        f"  first_to_last={m['first_to_last']:.4f} (joint {m['argmax_joint']})  "
        f"total_path={m['total_path']:.4f}  directionality={m['directionality']:.3f}  "
        f"grip_span={[f'{g:.4f}' for g in m['grip_span']]}"
    )
    return m["directionality"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5694)
    ap.add_argument("--instruction", default="fold towel")
    ap.add_argument("--samples", type=int, default=5, help="How many times to query (shows sampling variance).")
    ap.add_argument("--unnorm_key", default=None, help="Leave unset to use the server default.")

    # Option 1: three still images (training order: high, left wrist, right wrist)
    ap.add_argument("--high")
    ap.add_argument("--left")
    ap.add_argument("--right")

    # Option 2: pull one in-distribution frame from each episode video
    ap.add_argument("--from-video", action="store_true", help="Treat --high/--left/--right as video paths.")
    ap.add_argument("--frame", type=int, default=0, help="Frame index to decode when --from-video.")

    # Ground-truth comparison: a [T,14] GROUPED chunk dumped to .npy from the
    # dataset at the SAME index you fed as the observation. See snippet in the
    # module docstring for how to extract it from a dataset sample.
    ap.add_argument("--gt-npy", help="Path to a [T,14] grouped GT action chunk (.npy) for comparison.")
    ap.add_argument("--from-dataset-index", type=int, default=None,
                    help="(used by your dataset-loading branch; referenced only for the GT verdict message)")
    args = ap.parse_args()

    if not (args.high and args.left and args.right):
        ap.error("Provide --high, --left and --right (images, or videos with --from-video).")

    if args.from_video:
        images = [
            frame_from_video(args.high, args.frame),
            frame_from_video(args.left, args.frame),
            frame_from_video(args.right, args.frame),
        ]
    else:
        images = [load_image(args.high), load_image(args.left), load_image(args.right)]

    client = WebsocketClientPolicy(host=args.host, port=args.port)
    print("server metadata:", client.get_server_metadata())

    request: dict = {"examples": [{"image": images, "lang": args.instruction}], "do_sample": False}
    if args.unnorm_key:
        request["unnorm_key"] = args.unnorm_key

    f2l: List[float] = []
    dirs: List[float] = []
    print(f"\nprobing {args.samples}x  instruction={args.instruction!r}")
    print("-" * 78)
    for i in range(args.samples):
        chunk = parse_actions(client.predict_action(request))
        m = chunk_metrics(chunk)
        f2l.append(m["first_to_last"])
        dirs.append(m["directionality"])
        print(
            f"sample {i}: shape={chunk.shape}  first_to_last={m['first_to_last']:.4f} rad "
            f"(joint {m['argmax_joint']})  step_to_step={m['step_to_step']:.4f}  "
            f"total_path={m['total_path']:.4f}  dir={m['directionality']:.3f}  "
            f"grip_span={[f'{g:.4f}' for g in m['grip_span']]}"
        )
    client.close()

    # Optional ground-truth comparison: pass a [T,14] grouped chunk as .npy.
    gt_dir: Optional[float] = None
    if args.gt_npy:
        gt_dir = report_gt_chunk(np.load(args.gt_npy))

    f2l_arr = np.asarray(f2l)
    dir_arr = np.asarray(dirs)
    print("-" * 78)
    print(f"first_to_last over {args.samples} samples: mean={f2l_arr.mean():.4f}  std={f2l_arr.std():.4f}  max={f2l_arr.max():.4f} rad")
    print(f"directionality  over {args.samples} samples: mean={dir_arr.mean():.3f}  (>~0.4 reach, <~0.15 jitter)")
    if gt_dir is not None:
        print("-" * 78)
        if gt_dir > 0.4 and dir_arr.mean() < 0.15:
            print(f"  VERDICT: GT directionality={gt_dir:.3f} (real reach) but prediction={dir_arr.mean():.3f} (jitter).")
            print("           => case (B) CONFIRMED. The demo at this index moves with intent;")
            print("              the checkpoint emits a directionless chunk. Look upstream:")
            print("              normalization stats / convergence / abs-qpos embodiment frame.")
        elif gt_dir < 0.15:
            print(f"  VERDICT: GT directionality={gt_dir:.3f} is ALSO low -> index {args.from_dataset_index} is a HOLD")
            print("           frame in the demo. Re-run on a mid-reach index before concluding.")
        else:
            print(f"  GT directionality={gt_dir:.3f}, prediction={dir_arr.mean():.3f} -- inspect manually.")
    print("\ninterpretation:")
    if f2l_arr.mean() < 0.08:
        print("  * Net intended motion is tiny (< ~0.08 rad) even on this fixed frame.")
        print("    If this frame is a TRAINING frame -> case (B): the checkpoint outputs 'hold'.")
        print("    Look at data quality / normalization stats / convergence / embodiment frame,")
        print("    NOT the action-chunk length.")
        print("  * If this frame is a LIVE frame -> compare against a training frame next.")
    else:
        print("  * The policy DOES intend real motion on this frame (good).")
        print("    If live rollout is static but this isn't -> case (A): live observation is OOD.")
        print("    Diff this frame against your live /tmp dump (camera order/content/color).")
    if f2l_arr.std() >= 0.5 * max(f2l_arr.mean(), 1e-6):
        print("  * NOTE: high sample-to-sample variance => chunk is mostly sampling noise,")
        print("    consistent with the head having no strong conditioning signal.")


if __name__ == "__main__":
    main()
