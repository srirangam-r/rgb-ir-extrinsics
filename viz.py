
import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def visualize_matches(img0, img1, mkpts0, mkpts1):
    """
    Create a side-by-side visualization of matches and display inline.
    img0, img1: grayscale float32 [0,1] or uint8
    mkpts0, mkpts1: N x 2 in (x, y) pixel coordinates
    """
    # Convert grayscale float32 [0,1] to uint8 BGR
    def to_uint8_bgr(im):
        if im.dtype != np.uint8:
            im = (im * 255.0).clip(0, 255).astype(np.uint8)
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        return im

    im0 = to_uint8_bgr(img0)
    im1 = to_uint8_bgr(img1)

    h0, w0 = im0.shape[:2]
    h1, w1 = im1.shape[:2]
    h = max(h0, h1)
    w = w0 + w1

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:h0, :w0] = im0
    canvas[:h1, w0:w0 + w1] = im1

    # Draw matches: lines transparent, dots opaque
    overlay = canvas.copy()
    alpha = 0.5
    
    # 1. Draw lines on overlay
    for p0, p1 in zip(mkpts0, mkpts1):
        x0, y0 = int(p0[0]), int(p0[1])
        x1, y1 = int(p1[0] + w0), int(p1[1])
        color = (0, 255, 0)
        cv2.line(overlay, (x0, y0), (x1, y1), color, 1)
        
    # 2. Blend lines
    canvas = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)

    # 3. Draw dots on top (opaque)
    for p0, p1 in zip(mkpts0, mkpts1):
        x0, y0 = int(p0[0]), int(p0[1])
        x1, y1 = int(p1[0] + w0), int(p1[1])
        color = (0, 255, 0)
        cv2.circle(canvas, (x0, y0), 2, color, -1)
        cv2.circle(canvas, (x1, y1), 2, color, -1)

    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    return canvas

def show_epipolar_lines(img0, img1, pts0, pts1, F, sample_step=5):
    """
    img0, img1: grayscale uint8 or float images (same coordinate system as pts0, pts1).
    pts0, pts1: Nx2 inlier points (float).
    F: 3x3 fundamental matrix mapping img0 -> img1.
    sample_step: draw every Nth match for clarity.
    """
    def to_uint8(im):
        if im.dtype != np.uint8:
            im = (im * 255.0).clip(0, 255).astype(np.uint8)
        return im

    img0_u8 = to_uint8(img0)
    img1_u8 = to_uint8(img1)

    # OpenCV expects (N,1,2)
    pts0_reshaped = pts0.reshape(-1, 1, 2).astype(np.float32)
    pts1_reshaped = pts1.reshape(-1, 1, 2).astype(np.float32)

    # Lines in image 1 corresponding to pts0 in image 0
    lines1 = cv2.computeCorrespondEpilines(pts0_reshaped, 1, F)  # from img0 to img1
    lines1 = lines1.reshape(-1, 3)

    # Lines in image 0 corresponding to pts1 in image 1
    lines0 = cv2.computeCorrespondEpilines(pts1_reshaped, 2, F)  # from img1 to img0
    lines0 = lines0.reshape(-1, 3)

    img0_with_lines = _draw_lines_on_image(img0_u8, lines0, pts0, sample_step=sample_step)
    img1_with_lines = _draw_lines_on_image(img1_u8, lines1, pts1, sample_step=sample_step)

    # Show side-by-side inline
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epipolar lines in Image 0")
    plt.imshow(cv2.cvtColor(img0_with_lines, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Epipolar lines in Image 1")
    plt.imshow(cv2.cvtColor(img1_with_lines, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.show()

    return img0_with_lines, img1_with_lines

def _draw_lines_on_image(img, lines, pts, sample_step=1):
    if img.ndim == 2:
        color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        color_img = img.copy()

    h, w = color_img.shape[:2]

    for i in range(0, len(lines), sample_step):
        a, b, c = lines[i]
        if abs(b) < 1e-6:
            continue

        x0, y0 = 0, int(-c / b)
        x1, y1 = w, int(-(c + a * w) / b)

        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.line(color_img, (x0, y0), (x1, y1), color, 1)

        x, y = pts[i].astype(int)
        cv2.circle(color_img, (x, y), 3, color, -1)

    return color_img

def plot_depth_map(depth_map, title="Estimated Depth Map"):
    plt.figure(figsize=(10, 6))
    plt.imshow(depth_map, cmap='plasma')
    plt.colorbar(label='Depth (meters)')
    plt.title(title)
    plt.axis('off')
    plt.show()

def backproject_depth(depth_map, intrinsics):
    """
    Converts a 2D depth map into a 3D point cloud (N, 3).
    
    Args:
        depth_map: (H, W) numpy array of depth values (meters).
        intrinsics: Dict containing 'fx', 'fy', 'cx1', 'cy' (or 'cx').
    
    Returns:
        pts3d: (N, 3) numpy array of [x, y, z] coordinates.
    """
    H, W = depth_map.shape
    
    # 1. Create pixel grid
    v_grid, u_grid = np.indices((H, W))
    
    # 2. Flatten everything
    u = u_grid.flatten()
    v = v_grid.flatten()
    z = depth_map.flatten()
    
    # 3. Get Parameters
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    # Handle key naming variation (cx1 vs cx)
    cx = intrinsics.get('cx1', intrinsics.get('cx', 0))
    cy = intrinsics.get('cy', 0)
    
    # 4. Back-project
    # x = (u - cx) * z / fx
    # y = (v - cy) * z / fy
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Stack into (N, 3)
    pts3d = np.stack([x, y, z], axis=1)
    
    return pts3d

def plot_point_cloud(pts3d, color_input=None, min_z=0.1, max_z=10.0, subsample=5, title="Point Cloud"):
    """
    Visualizes a 3D point cloud using Plotly.
    
    Args:
        pts3d: (N, 3) numpy array of X, Y, Z coordinates.
        color_input: Optional. Can be:
                     - (H, W, 3) Image (will be flattened automatically).
                     - (N, 3) Array of RGB values.
        min_z, max_z: Depth clipping range (meters).
        subsample: Display every Nth point (higher = faster).
    """
    # 1. Handle Colors
    if color_input is None:
        # Default to coloring by Depth (Z)
        colors_flat = pts3d[:, 2] 
        use_colorscale = True
    else:
        # If input is an Image (H, W, 3), flatten it
        if color_input.ndim == 3:
            color_input = cv2.cvtColor(color_input, cv2.COLOR_BGR2RGB) # Ensure RGB
            colors_flat = color_input.reshape(-1, 3)
        else:
            colors_flat = color_input
            
        # Normalize to [0, 255] uint8 for Plotly string formatting
        if colors_flat.max() <= 1.0:
            colors_flat = (colors_flat * 255).astype(np.uint8)
        use_colorscale = False

    # 2. Filter by Depth (Z)
    mask = (pts3d[:, 2] > min_z) & (pts3d[:, 2] < max_z)
    
    pts_filtered = pts3d[mask][::subsample]
    
    if len(pts_filtered) == 0:
        print("Warning: No points found in valid Z-range!")
        return

    # 3. Process Colors for filtered points
    if use_colorscale:
        marker_color = pts_filtered[:, 2] # Use Z values
        colorscale = 'Magma'
    else:
        # Subset colors and convert to CSS strings 'rgb(r,g,b)'
        col_filtered = colors_flat[mask][::subsample]
        marker_color = [f'rgb({r},{g},{b})' for r, g, b in col_filtered]
        colorscale = None

    print(f"[{title}] Displaying {len(pts_filtered)} points (Original: {len(pts3d)})")

    # 4. Create Plot
    fig = go.Figure(data=[go.Scatter3d(
        x=pts_filtered[:, 0],
        y=pts_filtered[:, 1],
        z=pts_filtered[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=marker_color,
            colorscale=colorscale,
            opacity=1.0
        )
    )])

    # 5. Layout Settings (Fixed 'yaxis' bug here)
    fig.update_layout(
        title=title,
        width=900,
        height=600,
        scene=dict(
            xaxis_title="X (Right)",
            yaxis_title="Y (Down)",
            zaxis_title="Z (Forward)",
            aspectmode='data',
            yaxis=dict(autorange="reversed") # Correct way to flip Y in Plotly
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()

def visualize_pnp_reprojection(img_path, object_points, image_points, rvec, tvec, K, D, inliers=None):
    """
    Visualizes the quality of the PnP solution.
    """
    # 1. Load Image
    if isinstance(img_path, str):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not load {img_path}")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = img_path.copy()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # 2. Filter Inliers (if provided)
    if inliers is not None:
        inliers = inliers.flatten()
        obj_pts = object_points[inliers]
        img_pts = image_points[inliers]
    else:
        obj_pts = object_points
        img_pts = image_points

    # 3. Project 3D points to 2D
    projected_points, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)
    projected_points = projected_points.squeeze()

    # 4. Draw
    # Create a high-res canvas for cleaner drawing if needed, but drawing on img is fine
    canvas = img.copy()
    
    # Calculate errors for title
    errors = np.linalg.norm(img_pts - projected_points, axis=1)
    mean_err = np.mean(errors)
    
    print(f"Visualizing {len(obj_pts)} inliers.")
    print(f"Mean Reprojection Error: {mean_err:.4f} pixels")

    for i, (pt_true, pt_pred) in enumerate(zip(img_pts, projected_points)):
        # True Point (Green Circle)
        cv2.circle(canvas, (int(pt_true[0]), int(pt_true[1])), 4, (0, 255, 0), 1)
        
        # Projected Point (Red Cross)
        cv2.drawMarker(canvas, (int(pt_pred[0]), int(pt_pred[1])), (255, 0, 0), 
                       markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1)
        
        # Error Line (Yellow)
        cv2.line(canvas, (int(pt_true[0]), int(pt_true[1])), 
                 (int(pt_pred[0]), int(pt_pred[1])), (255, 255, 0), 1)

    # 5. Plot
    plt.figure(figsize=(12, 8))
    plt.imshow(canvas)
    plt.title(f"PnP Reprojection (Mean Error: {mean_err:.2f} px)\nGreen=Observed, Red=Projected")
    plt.axis('off')
    plt.show()