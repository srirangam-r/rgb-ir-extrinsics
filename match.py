import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import pytorch_lightning as pl

# Make sure repo root is on PYTHONPATH
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
sys.path.append(str(REPO_ROOT))

from src.config.default import get_cfg_defaults
from src.lightning.lightning_loftr import PL_LoFTR


def load_gray(path, resize=None):
    """
    Load image as grayscale float32 in [0,1].
    Works for RGB and 16-bit thermal (Boson).
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {path}")

    # If 3-channel, convert to gray
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float32)

    max_val = img.max()
    if max_val > 1.0:
        # Heuristic: 16-bit vs 8-bit
        if max_val > 255:
            img /= 65535.0
        else:
            img /= 255.0

    if resize is not None:
        img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_AREA)

    return img  # H x W, float32 in [0,1]


def build_matcher(cfg_path, ckpt_path, imgresize=None, thr=None):
    """
    Build PL_LoFTR in test mode, on CPU, METHOD=matchanything_eloftr.
    """
    config = get_cfg_defaults()
    config.merge_from_file(cfg_path)

    pl.seed_everything(config.TRAINER.SEED)

    # Use MatchAnything ELoFTR branch
    config.METHOD = "matchanything_eloftr"

    # Optional: ROPE/NPE resize if used in config
    if hasattr(config.LOFTR.COARSE, "ROPE") and config.LOFTR.COARSE.ROPE:
        assert config.DATASET.NPE_NAME is not None
    if getattr(config.DATASET, "NPE_NAME", None) is not None and imgresize is not None:
        config.LOFTR.COARSE.NPE = [832, 832, imgresize, imgresize]

    # Disable FP16 (CPU-only)
    config.LOFTR.FP16 = False

    # Coarse matching threshold override if provided
    if thr is not None:
        config.LOFTR.MATCH_COARSE.THR = thr

    pl_loftr = PL_LoFTR(config, pretrained_ckpt=ckpt_path, test_mode=True)
    matcher = pl_loftr.matcher
    matcher.eval()  # stay on CPU

    return matcher, config


def match_two_images(img0_path, img1_path, cfg_path, ckpt_path,
                     resize=None, thr=None, use_fp16=False):
    """
    Match a single image pair using MatchAnything ELoFTR (CPU).
    `use_fp16` is ignored (kept for API compatibility).
    """
    matcher, config = build_matcher(
        cfg_path=cfg_path,
        ckpt_path=ckpt_path,
        imgresize=resize,
        thr=thr,
    )

    # Load images
    img0 = load_gray(img0_path, resize=resize)
    img1 = load_gray(img1_path, resize=resize)

    # Build batch for LoFTR: [B, 1, H, W] on CPU
    img0_t = torch.from_numpy(img0)[None, None].float()
    img1_t = torch.from_numpy(img1)[None, None].float()

    batch = {
        "image0": img0_t,
        "image1": img1_t,
    }

    # Forward (CPU, no autocast)
    with torch.no_grad():
        matcher(batch)

    mkpts0 = batch["mkpts0_f"].cpu().numpy()  # (N, 2)
    mkpts1 = batch["mkpts1_f"].cpu().numpy()  # (N, 2)
    mconf = batch["mconf"].cpu().numpy()      # (N,)

    return mkpts0, mkpts1, mconf, img0, img1


def visualize_matches(img0, img1, mkpts0, mkpts1, out_path="matches_vis.png"):
    """
    Create a side-by-side visualization of matches and save to out_path.
    img0, img1: grayscale float32 [0,1]
    mkpts0, mkpts1: N x 2 in (x, y) pixel coordinates
    """
    # Convert grayscale float32 [0,1] to uint8 BGR
    im0 = (img0 * 255.0).clip(0, 255).astype(np.uint8)
    im1 = (img1 * 255.0).clip(0, 255).astype(np.uint8)

    im0 = cv2.cvtColor(im0, cv2.COLOR_GRAY2BGR)
    im1 = cv2.cvtColor(im1, cv2.COLOR_GRAY2BGR)

    h0, w0 = im0.shape[:2]
    h1, w1 = im1.shape[:2]
    h = max(h0, h1)
    w = w0 + w1

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:h0, :w0] = im0
    canvas[:h1, w0:w0 + w1] = im1

    # Draw matches
    for p0, p1 in zip(mkpts0, mkpts1):
        x0, y0 = p0
        x1, y1 = p1
        color = (0, 255, 0)
        cv2.circle(canvas, (int(x0), int(y0)), 2, color, -1)
        cv2.circle(canvas, (int(x1 + w0), int(y1)), 2, color, -1)
        cv2.line(canvas, (int(x0), int(y0)), (int(x1 + w0), int(y1)), color, 1)

    cv2.imwrite(out_path, canvas)
    print(f"Saved visualization to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Match a single image pair using LoFTR-style matcher."
    )
    parser.add_argument(
        "--cfg",
        default="configs/models/eloftr_model.py",
        help="Main config path (yaml/py)"
    )
    parser.add_argument(
        "--ckpt",
        default="weights/matchanything_eloftr.ckpt",
        help="Checkpoint path"
    )
    parser.add_argument(
        "--img0",
        default="imgs/rgb.png",
        help="Path to first image (e.g. RGB)"
    )
    parser.add_argument(
        "--img1",
        default="imgs/ir.png",
        help="Path to second image (e.g. IR)"
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=640,
        help="Optional square resize (e.g. 640). If None, keep original."
    )
    parser.add_argument(
        "--thr",
        type=float,
        default=None,
        help="Optional coarse matching threshold override."
    )
    parser.add_argument(
        "--no_fp16",
        action="store_true",
        help="Disable FP16/autocast, run pure FP32. (Ignored in CPU mode.)"
    )
    args = parser.parse_args()

    mkpts0, mkpts1, mconf, img0, img1 = match_two_images(
        img0_path=args.img0,
        img1_path=args.img1,
        cfg_path=args.cfg,
        ckpt_path=args.ckpt,
        resize=args.resize,
        thr=args.thr,
        use_fp16=not args.no_fp16,
    )

    print(f"#matches: {mkpts0.shape[0]}")
    print("First 10 matches (x0, y0) -> (x1, y1), conf:")
    for i in range(min(10, mkpts0.shape[0])):
        print(mkpts0[i], "->", mkpts1[i], "conf =", mconf[i])

    # Directly visualize in this script
    visualize_matches(img0, img1, mkpts0, mkpts1, out_path="matches_vis.png")


if __name__ == "__main__":
    main()
