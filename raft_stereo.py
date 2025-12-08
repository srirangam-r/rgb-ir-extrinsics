import sys
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation as R

RAFT_ROOT = Path("/home/nail/stuff/rgb-ir/rgb-ir-extrinsics/RAFT-Stereo").resolve()
sys.path.append(str(RAFT_ROOT))

try:
    from core.raft_stereo import RAFTStereo
    from core.utils.utils import InputPadder
except ImportError:
    print(f"Error: Could not import RAFT modules. Check if {RAFT_ROOT} is correct.")
    sys.exit(1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class StereoPreprocessor:
    def __init__(self, K_L, D_L, K_R, D_R, q_input, t_input, img_size):
        self.img_size = img_size
        self.K_L, self.D_L = K_L, D_L
        self.K_R, self.D_R = K_R, D_R
        
        # 1. Invert Extrinsics (Right-in-Left -> Left-to-Right)
        # Assuming your q_1_0/t_1_0 are Right-in-Left (Cam 1 in Cam 0)
        R_R2L = R.from_quat(q_input).as_matrix()
        R_L2R = R_R2L
        T_L2R = t_input

        # 2. Compute Base Rectification
        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            K_L, D_L, K_R, D_R, img_size, R_L2R, T_L2R,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )
        
        self.y_shift = 0.0
        self.update_maps()

    def update_maps(self):
        """Regenerates maps with optional vertical shift."""
        P2_shifted = self.P2.copy()
        P2_shifted[1, 2] += self.y_shift 
        
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(self.K_L, self.D_L, self.R1, self.P1, self.img_size, cv2.CV_32FC1)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(self.K_R, self.D_R, self.R2, P2_shifted, self.img_size, cv2.CV_32FC1)

        self.intrinsics = {
            'fx': self.P1[0, 0], 'fy': self.P1[1, 1], 
            'cx1': self.P1[0, 2], 'cx2': P2_shifted[0, 2], 'cy': self.P1[1, 2]
        }
        self.extrinsics = {'baseline': abs(self.P2[0, 3] / self.P1[0, 0])}

    def rectify_images(self, left_img, right_img):
        return (cv2.remap(left_img, self.map1x, self.map1y, cv2.INTER_LINEAR),
                cv2.remap(right_img, self.map2x, self.map2y, cv2.INTER_LINEAR))

    def auto_align(self, imgL, imgR):
        """Automatically corrects vertical misalignment."""
        print("[Preprocessor] Running Auto-Alignment...")
        # Get initial rectification
        rL, rR = self.rectify_images(imgL, imgR)
        
        # Detect features
        orb = cv2.ORB_create(3000)
        kp1, des1 = orb.detectAndCompute(rL, None)
        kp2, des2 = orb.detectAndCompute(rR, None)
        
        if des1 is None or des2 is None:
            print("[Preprocessor] Warning: No features found.")
            return

        # Match
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        # Filter for strong matches
        valid_dy = []
        for m in matches:
            ptL = kp1[m.queryIdx].pt
            ptR = kp2[m.trainIdx].pt
            dy = ptR[1] - ptL[1]
            if abs(dy) < 50: # Sanity check
                valid_dy.append(dy)
        
        if len(valid_dy) < 10:
            print("[Preprocessor] Warning: Not enough matches to align.")
            return

        # Apply Correction
        median_dy = np.median(valid_dy)
        print(f"[Preprocessor] Correction applied: {median_dy:.3f} pixels")
        self.y_shift -= median_dy 
        self.update_maps()

def estimate_stereo_depth(left_img, right_img, model, intrinsics, extrinsics):
    """
    Robust Depth Estimation.
    """
    img1 = torch.from_numpy(left_img).permute(2, 0, 1).float()[None].to(DEVICE)
    img2 = torch.from_numpy(right_img).permute(2, 0, 1).float()[None].to(DEVICE)
    
    padder = InputPadder(img1.shape, divis_by=32)
    img1, img2 = padder.pad(img1, img2)

    with torch.no_grad():
        _, flow_up = model(img1, img2, iters=32, test_mode=True)

    flow = flow_up.cpu().numpy().squeeze()
    flow = padder.unpad(torch.from_numpy(flow[None, None]))[0, 0].numpy()
    
    # Left->Right flow is negative. Disparity is -flow.
    disparity = -flow
    disparity[disparity < 0] = 0 # Filter noise
    
    # Depth Formula
    fx = intrinsics['fx']
    base = extrinsics['baseline']
    depth = (fx * base) / (disparity + 1e-6)
    
    return depth

def load_raft_model(model_path):
    args = type('Args', (), {
        'corr_implementation': 'alt', 'mixed_precision': True, 'shared_backbone': False,
        'corr_levels': 4, 'corr_radius': 4, 'n_downsample': 2, 'context_norm': 'batch',
        'slow_fast_gru': False, 'n_gru_layers': 3, 'hidden_dims': [128]*3
    })()
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(model_path))
    model = model.module
    model.to(DEVICE)
    model.eval()
    return model
