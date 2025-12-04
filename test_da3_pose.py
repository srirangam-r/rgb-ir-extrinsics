import sys
import os
import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# Add Depth-Anything-3 to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Depth-Anything-3/src'))

from depth_anything_3.api import DepthAnything3

def run_test():
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img0_path = os.path.join(base_dir, 'imgs', 'rgb0.png')
    img1_path = os.path.join(base_dir, 'imgs', 'rgb1.png')

    # Load images
    img0 = cv2.imread(img0_path)
    img1 = cv2.imread(img1_path)
    
    if img0 is None or img1 is None:
        print("Error loading images")
        return

    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    # Intrinsics and Distortion
    K_rgb0 = np.array([
        [889.44722255,   0.0,         365.30215844],
        [  0.0,         890.67612482, 285.76081066],
        [  0.0,           0.0,         1.0        ]
    ], dtype=np.float64)

    d_rgb0 = np.array([-0.20743818, 0.14175898, 0.00112518, -0.00010317], dtype=np.float64)

    K_rgb1 = np.array([
        [889.81741800,   0.0,         375.16051052],
        [  0.0,         891.25501167, 277.33320298],
        [  0.0,           0.0,         1.0        ]
    ], dtype=np.float64)

    d_rgb1 = np.array([-0.20879479, 0.14454576, -0.00018156, 0.00030712], dtype=np.float64)

    # Undistort images and get new K
    # alpha=0 to crop to valid pixels
    h0, w0 = img0.shape[:2]
    new_K0, roi0 = cv2.getOptimalNewCameraMatrix(K_rgb0, d_rgb0, (w0, h0), 0, (w0, h0))
    img0_undistorted = cv2.undistort(img0, K_rgb0, d_rgb0, None, new_K0)
    x, y, w, h = roi0
    img0_undistorted = img0_undistorted[y:y+h, x:x+w]
    # Adjust K for crop
    new_K0[0, 2] -= x
    new_K0[1, 2] -= y

    h1, w1 = img1.shape[:2]
    new_K1, roi1 = cv2.getOptimalNewCameraMatrix(K_rgb1, d_rgb1, (w1, h1), 0, (w1, h1))
    img1_undistorted = cv2.undistort(img1, K_rgb1, d_rgb1, None, new_K1)
    x, y, w, h = roi1
    img1_undistorted = img1_undistorted[y:y+h, x:x+w]
    # Adjust K for crop
    new_K1[0, 2] -= x
    new_K1[1, 2] -= y

    # Extrinsics
    # q_1_0 = [x, y, z, w]
    q_1_0 = np.array([0.01308697, -0.0132229, -0.00458315, 0.99981642], dtype=np.float64)
    t_1_0 = np.array([0.32019253, 0.00262298, -0.00700941], dtype=np.float64)

    r = R.from_quat(q_1_0)
    R_1_0 = r.as_matrix()

    T_0 = np.eye(4)
    T_1 = np.eye(4)
    T_1[:3, :3] = R_1_0
    T_1[:3, 3] = t_1_0

    extrinsics = np.stack([T_0, T_1], axis=0) # (2, 4, 4)
    intrinsics = np.stack([new_K0, new_K1], axis=0) # (2, 3, 3)
    
    # Prepare input list
    images = [img0_undistorted, img1_undistorted]

    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_name = "depth-anything/DA3NESTED-GIANT-LARGE" 
    
    print(f"Loading model {model_name}...")
    model = DepthAnything3.from_pretrained(model_name)
    model = model.to(device=device)

    # Inference
    print("Running inference...")
    prediction = model.inference(
        images,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        use_ray_pose=False 
    )

    print("Inference done.")
    print("Depth shape:", prediction.depth.shape)
    
    # Save results
    np.savez(os.path.join(base_dir, 'da3_pose_output.npz'), depth=prediction.depth, extrinsics=prediction.extrinsics, intrinsics=prediction.intrinsics)
    
    # Visualize
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(prediction.depth[0])
    plt.title("Depth 0")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(prediction.depth[1])
    plt.title("Depth 1")
    plt.axis('off')
    
    plt.savefig(os.path.join(base_dir, 'da3_pose_depth.png'))
    print("Saved results to da3_pose_output.npz and da3_pose_depth.png")

if __name__ == "__main__":
    run_test()
