"""Aircraft PnP pose estimation."""
import sys, yaml, cv2, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def main():
    import argparse
    p = argparse.ArgumentParser(description='Aircraft PnP pose estimation')
    p.add_argument('--config', required=True,
                   help='Camera config (e.g. configs/cameras/camera_25mm_far.yaml)')
    p.add_argument('--aircraft-3d', default='configs/aircraft_points.yaml')
    p.add_argument('--aircraft-2d', required=True, help='Aircraft 2D annotation YAML')
    p.add_argument('--output', '-o', default='output/aircraft_pose.csv')
    args = p.parse_args()

    from sls_calib.config_validator import load_camera_config
    _, K, dist = load_camera_config(args.config)

    with open(args.aircraft_3d, encoding='utf-8') as f: ac3d = yaml.safe_load(f)

    # Validate coordinate system
    cs = ac3d.get('coordinate_system', '')
    if cs != 'B':
        raise ValueError(
            f'Aircraft PnP requires points in aircraft body frame (B), '
            f'got coordinate_system={cs!r}. '
            f'Run tools/convert_to_B_frame.py to convert from G to B frame.'
        )
    if ac3d.get('unit', 'mm') != 'mm':
        raise ValueError('Aircraft point unit must be mm')

    with open(args.aircraft_2d, encoding='utf-8') as f: ac2d = yaml.safe_load(f)

    # Support both 'points' and 'points_chinese' keys
    pts3d_all = ac3d.get('points', {})
    if ac3d.get('points_chinese'):
        pts3d_all = {**pts3d_all, **ac3d['points_chinese']}

    obj, img = [], []
    for name, info in pts3d_all.items():
        if name in ac2d.get('points', {}):
            p2d = ac2d['points'][name]
            px, py = float(p2d['pixel_x']), float(p2d['pixel_y'])
            if px < 0 or py < 0:
                continue
            x, y, z = float(info['x_mm']), float(info['y_mm']), float(info['z_mm'])
            obj.append([x, y, z])
            img.append([px, py])

    n_total = len(obj)
    if n_total < 6:  # need at least 6 visible points (plan requires 8)
        print(f'Insufficient correspondences: {n_total} (need >=6, recommend >=8)')
        sys.exit(1)

    obj_arr = np.array(obj, dtype=np.float64)
    img_arr = np.array(img, dtype=np.float64)

    # --- RANSAC with strict threshold ---
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_arr, img_arr, K, dist,
        flags=cv2.SOLVEPNP_EPNP, iterationsCount=500,
        reprojectionError=3.0, confidence=0.99)
    if not success:
        print('Aircraft PnP failed (RANSAC 3px)'); sys.exit(1)

    n_inl = len(inliers) if inliers is not None else 0
    inlier_ratio = n_inl / n_total if n_total > 0 else 0

    if n_inl < 6:
        print(f'Aircraft PnP failed: only {n_inl}/{n_total} inliers (need >=6)')
        sys.exit(1)
    if inlier_ratio < 0.75:
        print(f'Aircraft PnP failed: inlier ratio {inlier_ratio:.1%} < 75%')
        sys.exit(1)

    # --- LM refinement on inliers ---
    inl_idx = inliers.ravel()
    inl_mask = np.zeros(len(obj_arr), dtype=bool)
    inl_mask[inl_idx] = True
    rvec2, tvec2 = cv2.solvePnPRefineLM(
        obj_arr[inl_mask], img_arr[inl_mask], K, dist, rvec, tvec,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    rvec, tvec = rvec2, tvec2

    # --- Project ALL visible points (not just inliers) ---
    proj_all, _ = cv2.projectPoints(obj_arr, rvec, tvec, K, dist)
    errs_all = np.linalg.norm(proj_all.reshape(-1, 2) - img_arr, axis=1)
    rmse_all = float(np.sqrt(np.mean(errs_all**2)))
    max_err_all = float(np.max(errs_all))

    # --- Inlier-only RMSE ---
    proj_inl, _ = cv2.projectPoints(obj_arr[inl_mask], rvec, tvec, K, dist)
    errs_inl = np.linalg.norm(proj_inl.reshape(-1, 2) - img_arr[inl_mask], axis=1)
    rmse_inl = float(np.sqrt(np.mean(errs_inl**2)))

    # --- Quality flag ---
    if (rmse_all <= 1.0 and max_err_all <= 2.0 and
            n_inl >= 6 and inlier_ratio >= 0.85):
        quality = 'good'
    elif rmse_all <= 2.0 and max_err_all <= 5.0 and n_inl >= 5:
        quality = 'fair'
    else:
        quality = 'poor'

    R, _ = cv2.Rodrigues(rvec)
    print(f'Aircraft PnP: {n_inl}/{n_total} inliers ({inlier_ratio:.0%}), '
          f'all-RMSE={rmse_all:.3f}px inl-RMSE={rmse_inl:.3f}px '
          f'max={max_err_all:.1f}px quality={quality}')
    print(f'R:\n{R}')
    print(f't (mm): {tvec.ravel()}')
    print(f'Per-point errors (all): '
          f'{", ".join(f"{e:.1f}" for e in errs_all)}')

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    rv = rvec.ravel()
    tv = tvec.ravel()
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write('image_id,rvec_x,rvec_y,rvec_z,tvec_x,tvec_y,tvec_z,'
                'rmse_all_px,rmse_inl_px,max_err_px,inlier_count,'
                'total_count,quality\n')
        f.write(f'{Path(args.aircraft_2d).stem},{rv[0]:.6f},{rv[1]:.6f},'
                f'{rv[2]:.6f},{tv[0]:.4f},{tv[1]:.4f},{tv[2]:.4f},'
                f'{rmse_all:.4f},{rmse_inl:.4f},{max_err_all:.4f},'
                f'{n_inl},{n_total},{quality}\n')
    print(f'-> {args.output}')

    if quality == 'poor':
        print('WARNING: poor quality — point library or annotation may need review')
        sys.exit(1)

if __name__ == '__main__':
    main()
