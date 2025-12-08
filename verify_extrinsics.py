
import cv2
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

def get_matches(img0, img1):
    sift = cv2.SIFT_create()
    kp0, des0 = sift.detectAndCompute(img0, None)
    kp1, des1 = sift.detectAndCompute(img1, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des0, des1, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    pts0 = np.float32([kp0[m.queryIdx].pt for m in good])
    pts1 = np.float32([kp1[m.trainIdx].pt for m in good])
    
    # RANSAC to remove bad outliers for cleaner evaluation
    if len(good) > 8:
        _, mask = cv2.findFundamentalMat(pts0, pts1, cv2.FM_RANSAC, 1.0, 0.99)
        pts0 = pts0[mask.ravel() == 1]
        pts1 = pts1[mask.ravel() == 1]
    
    return pts0, pts1

def compute_epipolar_error(pts0, pts1, F):
    # Sampson distance
    # lines1 = F * pts0
    # lines0 = F.T * pts1
    
    pts0_h = np.hstack([pts0, np.ones((len(pts0), 1))])
    pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
    
    lines1 = (F @ pts0_h.T).T # N x 3
    lines0 = (F.T @ pts1_h.T).T # N x 3
    
    # constraint pts1^T * F * pts0
    constraint = np.sum(pts1_h * (F @ pts0_h.T).T, axis=1) # N
    
    # Sampson error
    denom = lines1[:,0]**2 + lines1[:,1]**2 + lines0[:,0]**2 + lines0[:,1]**2
    err = constraint**2 / denom
    return np.mean(err)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img0_path = os.path.join(base_dir, 'imgs', 'rgb0.png')
    img1_path = os.path.join(base_dir, 'imgs', 'rgb1.png')

    img0 = cv2.imread(img0_path)
    img1 = cv2.imread(img1_path)
    
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

    # Undistort images
    h0, w0 = img0.shape[:2]
    new_K0, roi0 = cv2.getOptimalNewCameraMatrix(K_rgb0, d_rgb0, (w0, h0), 0, (w0, h0))
    img0_undistorted = cv2.undistort(img0, K_rgb0, d_rgb0, None, new_K0)
    
    h1, w1 = img1.shape[:2]
    new_K1, roi1 = cv2.getOptimalNewCameraMatrix(K_rgb1, d_rgb1, (w1, h1), 0, (w1, h1))
    img1_undistorted = cv2.undistort(img1, K_rgb1, d_rgb1, None, new_K1)

    pts0, pts1 = get_matches(img0_undistorted, img1_undistorted)
    print(f"Number of inlier matches: {len(pts0)}")

    # Configurations to test
    # q_user = [-0.01308697,  0.0132229 ,  0.00458315,  0.99981642]
    # t_user = [ 0.32019253,  0.00262298, -0.00700941]
    
    configs = {
        "User_Provided": {
            "q": np.array([-0.01308697,  0.0132229 ,  0.00458315,  0.99981642]),
            "t": np.array([ 0.32019253,  0.00262298, -0.00700941])
        },
        "File_Provided (test_da3_pose.py)": {
            "q": np.array([0.01308697, -0.0132229, -0.00458315, 0.99981642]),
            "t": np.array([0.32019253, 0.00262298, -0.00700941])
        }
    }
    
    for name, conf in configs.items():
        q, t = conf["q"], conf["t"]
        
        # Test as transformation matrix T (World to Cam or Cam to World?)
        # Case 1: T_1_in_0 (Pose of cam1 in cam0 frame).
        # This implies X_0 = T * X_1.
        # To compute F, we need proj matrices P0=K[I|0], P1=K[R|t] where X1 = R*X0 + t.
        # So we need T_0_to_1 = inv(T_1_in_0).
        
        r_mat = R.from_quat(q).as_matrix()
        T_1_in_0 = np.eye(4)
        T_1_in_0[:3, :3] = r_mat
        T_1_in_0[:3, 3] = t
        
        T_0_to_1 = np.linalg.inv(T_1_in_0)
        R_rel = T_0_to_1[:3, :3]
        t_rel = T_0_to_1[:3, 3]
        
        # Fundamental Matrix
        # F = K1_inv_T * [t]_x * R * K0_inv
        
        t_x = np.array([
            [0, -t_rel[2], t_rel[1]],
            [t_rel[2], 0, -t_rel[0]],
            [-t_rel[1], t_rel[0], 0]
        ])
        
        E = t_x @ R_rel
        F = np.linalg.inv(new_K1).T @ E @ np.linalg.inv(new_K0)
        
        err = compute_epipolar_error(pts0, pts1, F)
        print(f"Config: {name} (as Pose 1_in_0)")
        print(f"  Mean Sampson Error: {err:.4f}")

        # Case 2: Treat q, t as the relative transform directly? Usually not "extrinsics" definition but worth checking.
        R_dir = r_mat
        t_dir = t
        t_x_dir = np.array([
            [0, -t_dir[2], t_dir[1]],
            [t_dir[2], 0, -t_dir[0]],
            [-t_dir[1], t_dir[0], 0]
        ])
        E_dir = t_x_dir @ R_dir
        F_dir = np.linalg.inv(new_K1).T @ E_dir @ np.linalg.inv(new_K0)
        err_dir = compute_epipolar_error(pts0, pts1, F_dir)
        print(f"Config: {name} (as Relative Transform 0->1)")
        print(f"  Mean Sampson Error: {err_dir:.4f}")
        print("-" * 30)

if __name__ == "__main__":
    main()
