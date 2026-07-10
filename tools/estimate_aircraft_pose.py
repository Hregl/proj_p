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

    if len(obj) < 4:
        print(f'Insufficient correspondences: {len(obj)} (need >=4)'); sys.exit(1)

    obj_arr = np.array(obj, dtype=np.float64)
    img_arr = np.array(img, dtype=np.float64)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_arr, img_arr, K, dist,
        flags=cv2.SOLVEPNP_EPNP, iterationsCount=500,
        reprojectionError=8.0, confidence=0.99)
    if not success:
        # Fallback: ITERATIVE with looser threshold
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_arr, img_arr, K, dist,
            flags=cv2.SOLVEPNP_ITERATIVE, iterationsCount=500,
            reprojectionError=15.0, confidence=0.99)
    if not success:
        print('Aircraft PnP failed'); sys.exit(1)

    R, _ = cv2.Rodrigues(rvec)
    n_inl = len(inliers) if inliers is not None else 0

    # 重投影误差(仅内点)
    if inliers is not None and len(inliers) > 0:
        inl_idx = inliers.ravel()
        mask = np.zeros(len(obj_arr), dtype=bool)
        mask[inl_idx] = True
        proj, _ = cv2.projectPoints(obj_arr[mask], rvec, tvec, K, dist)
        errs = np.linalg.norm(proj.reshape(-1, 2) - img_arr[mask], axis=1)
    else:
        proj, _ = cv2.projectPoints(obj_arr, rvec, tvec, K, dist)
        errs = np.linalg.norm(proj.reshape(-1, 2) - img_arr, axis=1)
    rmse = float(np.sqrt(np.mean(errs**2)))

    print(f'Aircraft PnP: {n_inl}/{len(obj)} inliers, RMSE={rmse:.3f}px')
    print(f'R:\n{R}')
    print(f't (mm): {tvec.ravel()}')

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    rv = rvec.ravel()
    tv = tvec.ravel()
    with open(args.output, 'w') as f:
        f.write('image_id,rvec_x,rvec_y,rvec_z,tvec_x,tvec_y,tvec_z,rmse_px,inlier_count\n')
        f.write(f'{Path(args.aircraft_2d).stem},{rv[0]:.6f},{rv[1]:.6f},{rv[2]:.6f},')
        f.write(f'{tv[0]:.4f},{tv[1]:.4f},{tv[2]:.4f},{rmse:.4f},{n_inl}\n')
    print(f'-> {args.output}')

if __name__ == '__main__':
    main()
