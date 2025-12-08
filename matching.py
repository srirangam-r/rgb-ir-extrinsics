
import cv2
import numpy as np
import torch
from pathlib import Path
from MatchAnything.imcui.ui.utils import get_matcher_zoo, load_config, DEVICE
from MatchAnything.imcui.api import ImageMatchingAPI

CONFIG_PATH = "MatchAnything/config/config.yaml"

def build_roma_matcher(config_path=CONFIG_PATH):
    """
    Build ROMA matcher via ImageMatchingAPI.
    """
    config = load_config(config_path)
    matcher_zoo = get_matcher_zoo(config["matcher_zoo"])

    if "matchanything_roma" not in matcher_zoo:
        raise RuntimeError("ROMA is not enabled in config/config.yaml")

    api = ImageMatchingAPI(conf=matcher_zoo["matchanything_roma"], device=DEVICE)
    return api

def limit_inliers(pred, max_inliers=800):
    """
    Limit number of inlier matches AFTER RANSAC.
    """
    if "mmkeypoints0_orig" not in pred:
        print("[WARN] No inlier matches (mmkeypoints*_orig) found in prediction.")
        return pred

    mk0 = pred["mmkeypoints0_orig"]
    mk1 = pred["mmkeypoints1_orig"]
    mconf = pred["mmconf"]

    if len(mk0) > max_inliers:
        print(f"[INFO] Limiting inliers: {len(mk0)} -> {max_inliers}")
        pred["mmkeypoints0_orig"] = mk0[:max_inliers]
        pred["mmkeypoints1_orig"] = mk1[:max_inliers]
        pred["mmconf"] = mconf[:max_inliers]
    else:
        print(f"[INFO] Inliers OK: {len(mk0)} (limit={max_inliers})")

    return pred

def match_two_images_roma(api, img_rgb_path, img_ir_path, max_inliers=800):
    """
    Match a single RGB-IR pair using MatchAnything ROMA.
    Returns: mkpts_rgb, mkpts_ir, mconf, rgb_gray, ir_gray
    """
    # Load RGB & IR (as color for ROMA)
    img_rgb = cv2.imread(str(img_rgb_path))  # BGR
    img_ir = cv2.imread(str(img_ir_path))    # BGR or grayscale, treated as color

    if img_rgb is None:
        raise FileNotFoundError(f"Could not read RGB image at {img_rgb_path}")
    if img_ir is None:
        raise FileNotFoundError(f"Could not read IR image at {img_ir_path}")

    # Convert BGR->RGB because MatchAnything expects RGB
    img_rgb_rgb = img_rgb[:, :, ::1][:, :, ::-1]
    img_ir_rgb = img_ir[:, :, ::1][:, :, ::-1]

    # Run ROMA
    pred = api(img_rgb_rgb, img_ir_rgb)

    # Use inliers if available
    if "mmkeypoints0_orig" in pred:
        print("Using inlier keypoints (mmkeypoints*_orig).")
        pred = limit_inliers(pred, max_inliers=max_inliers)
        mkpts_rgb = pred["mmkeypoints0_orig"]
        mkpts_ir = pred["mmkeypoints1_orig"]
        mconf = pred["mmconf"]
    else:
        print("Using raw keypoints (keypoints0/1).")
        mkpts_rgb = pred["keypoints0"]
        mkpts_ir = pred["keypoints1"]
        mconf = pred.get("confidence")

    # Convert to numpy arrays
    mkpts_rgb = np.asarray(mkpts_rgb, dtype=np.float32)
    mkpts_ir = np.asarray(mkpts_ir, dtype=np.float32)
    
    if mconf is not None:
        mconf = np.asarray(mconf, dtype=np.float32)
    else:
        mconf = np.ones(len(mkpts_rgb), dtype=np.float32)

    # Make grayscale float32 [0,1] copies for visualization/epipolar
    rgb_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    if img_ir.ndim == 2:
        ir_gray = img_ir.astype(np.float32) / 255.0
    else:
        ir_gray = cv2.cvtColor(img_ir, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    return mkpts_rgb, mkpts_ir, mconf, rgb_gray, ir_gray

def compute_epipolar_error_sampson(pts0, pts1, F):
    """
    Compute Sampson Error for the Fundamental Matrix.
    pts0: (N, 2)
    pts1: (N, 2)
    F: (3, 3)
    """
    pts0_h = np.hstack([pts0, np.ones((len(pts0), 1))])
    pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
    
    lines1 = (F @ pts0_h.T).T # N x 3
    lines0 = (F.T @ pts1_h.T).T # N x 3
    
    # constraint x'Fx
    constraint = np.sum(pts1_h * (F @ pts0_h.T).T, axis=1) # N
    
    # Sampson error
    denom = lines1[:,0]**2 + lines1[:,1]**2 + lines0[:,0]**2 + lines0[:,1]**2
    err = constraint**2 / (denom + 1e-8)
    return np.mean(err)
