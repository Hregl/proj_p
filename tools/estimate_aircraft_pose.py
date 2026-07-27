"""Stage 4: Aircraft PnP pose estimation (C_T_B).

Estimates the aircraft body frame (B) pose relative to the camera (C)
from 2D/3D aircraft marker correspondences.

Outputs per-point residuals, inlier/outlier status, and a quality flag.
"""

import sys, yaml, cv2, numpy as np, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    import argparse
    p = argparse.ArgumentParser(
        description='Stage 4: Aircraft PnP pose estimation (C_T_B)')
    p.add_argument('--config', required=True,
                   help='Camera config (e.g. configs/cameras/camera_25mm.yaml)')
    p.add_argument('--aircraft-3d', default='configs/aircraft_points_B.yaml',
                   help='Aircraft 3D points in B-frame (YAML)')
    p.add_argument('--aircraft-2d', required=True,
                   help='Aircraft 2D annotation YAML')
    p.add_argument('--output', '-o', default='output/aircraft_pose.csv')
    p.add_argument('--output-json', default=None,
                   help='Optional JSON output with per-point residuals')
    args = p.parse_args()

    from sls_calib.config_validator import load_camera_config
    _, K, dist = load_camera_config(args.config)

    # --- Load 3D points (B-frame) ---
    with open(args.aircraft_3d, encoding='utf-8') as f:
        ac3d = yaml.safe_load(f)

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

    # --- Load 2D annotations ---
    with open(args.aircraft_2d, encoding='utf-8') as f:
        ac2d = yaml.safe_load(f)

    # Support both 'points' and 'points_chinese' keys
    pts3d_all = ac3d.get('points', {})
    if ac3d.get('points_chinese'):
        pts3d_all = {**pts3d_all, **ac3d['points_chinese']}

    # --- Build correspondences ---
    obj, img, matched_names = [], [], []
    unmatched = []
    for name, info in pts3d_all.items():
        if name in ac2d.get('points', {}):
            p2d = ac2d['points'][name]
            px, py = float(p2d['pixel_x']), float(p2d['pixel_y'])
            if px < 0 or py < 0:
                continue
            x, y, z = float(info['x_mm']), float(info['y_mm']), float(info['z_mm'])
            obj.append([x, y, z])
            img.append([px, py])
            matched_names.append(name)
        else:
            unmatched.append(name)

    n_total = len(obj)
    if n_total < 6:
        print(f'FAIL: Insufficient correspondences: {n_total} (need >=6, '
              f'recommend >=8)')
        if unmatched:
            print(f'  Unmatched 3D points (not in 2D annotation): {unmatched}')
        sys.exit(1)

    obj_arr = np.array(obj, dtype=np.float64)
    img_arr = np.array(img, dtype=np.float64)

    # --- RANSAC with strict threshold ---
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_arr, img_arr, K, dist,
        flags=cv2.SOLVEPNP_EPNP, iterationsCount=500,
        reprojectionError=3.0, confidence=0.99)

    failure_reason = ''
    if not success:
        failure_reason = 'RANSAC failed (no solution found)'
        print(f'FAIL: {failure_reason}')
        sys.exit(1)

    n_inl = len(inliers) if inliers is not None else 0
    inlier_ratio = n_inl / n_total if n_total > 0 else 0

    if n_inl < 6:
        failure_reason = (f'Only {n_inl}/{n_total} inliers (need >=6). '
                          f'Check annotations and point library.')
        print(f'FAIL: {failure_reason}')
        sys.exit(1)

    if inlier_ratio < 0.75:
        failure_reason = (f'Inlier ratio {inlier_ratio:.1%} < 75%. '
                          f'Possible wrong point correspondences.')
        print(f'FAIL: {failure_reason}')
        sys.exit(1)

    # --- LM refinement on inliers ---
    inl_idx = inliers.ravel()
    inl_mask = np.zeros(len(obj_arr), dtype=bool)
    inl_mask[inl_idx] = True
    try:
        rvec2, tvec2 = cv2.solvePnPRefineLM(
            obj_arr[inl_mask], img_arr[inl_mask], K, dist, rvec, tvec,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
        rvec, tvec = rvec2, tvec2
    except cv2.error:
        pass  # Keep RANSAC result if LM fails

    # --- Compute per-point residuals ---
    proj_all, _ = cv2.projectPoints(obj_arr, rvec, tvec, K, dist)
    per_point_errors = {}
    inlier_status = {}
    for i, name in enumerate(matched_names):
        err = float(np.linalg.norm(proj_all[i, 0] - img_arr[i]))
        per_point_errors[name] = round(err, 3)
        inlier_status[name] = bool(inl_mask[i])

    errs_all = np.array(list(per_point_errors.values()))
    rmse_all = float(np.sqrt(np.mean(errs_all ** 2)))
    max_err_all = float(np.max(errs_all))

    # Inlier-only RMSE
    errs_inl = np.array([
        e for i, e in enumerate(per_point_errors.values())
        if inlier_status[matched_names[i]]])
    rmse_inl = float(np.sqrt(np.mean(errs_inl ** 2))) if len(errs_inl) > 0 else 999.0

    # --- Quality flag ---
    if (rmse_all <= 1.0 and max_err_all <= 2.0 and
            n_inl >= 6 and inlier_ratio >= 0.85):
        quality = 'good'
    elif rmse_all <= 2.0 and max_err_all <= 5.0 and n_inl >= 5:
        quality = 'fair'
    else:
        quality = 'poor'

    # --- Output ---
    R, _ = cv2.Rodrigues(rvec)
    print(f'Aircraft PnP: {n_inl}/{n_total} inliers ({inlier_ratio:.0%}), '
          f'all-RMSE={rmse_all:.3f}px inl-RMSE={rmse_inl:.3f}px '
          f'max={max_err_all:.1f}px quality={quality}')
    print(f'R:\n{R}')
    print(f't (mm): {tvec.ravel()}')
    print(f'Per-point errors (all):')
    for name in matched_names:
        status = 'INL' if inlier_status[name] else 'OUT'
        print(f'  {status} {name}: {per_point_errors[name]:.2f} px')

    # --- Save CSV ---
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    rv = rvec.ravel()
    tv = tvec.ravel()
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write('image_id,rvec_x,rvec_y,rvec_z,tvec_x,tvec_y,tvec_z,'
                'rmse_all_px,rmse_inl_px,max_err_px,inlier_count,'
                'total_count,quality,failure_reason\n')
        f.write(f'{Path(args.aircraft_2d).stem},{rv[0]:.6f},{rv[1]:.6f},'
                f'{rv[2]:.6f},{tv[0]:.4f},{tv[1]:.4f},{tv[2]:.4f},'
                f'{rmse_all:.4f},{rmse_inl:.4f},{max_err_all:.4f},'
                f'{n_inl},{n_total},{quality},'
                f'"{failure_reason}"\n')
    print(f'-> {args.output}')

    # --- Save JSON (optional) ---
    if args.output_json:
        json_data = {
            'image_id': Path(args.aircraft_2d).stem,
            'quality': quality,
            'n_total': n_total,
            'n_inliers': n_inl,
            'inlier_ratio': round(inlier_ratio, 4),
            'rmse_all_px': round(rmse_all, 4),
            'rmse_inl_px': round(rmse_inl, 4),
            'max_error_px': round(max_err_all, 4),
            'C_R_B': [[round(float(v), 6) for v in row] for row in R],
            'C_t_B_mm': [round(float(v), 4) for v in tv],
            'failure_reason': failure_reason,
            'per_point': {
                name: {
                    'error_px': per_point_errors[name],
                    'inlier': inlier_status[name],
                }
                for name in matched_names
            },
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f'-> {args.output_json}')

    if quality == 'poor':
        print('WARNING: poor quality — point library or annotation may need review')
        sys.exit(1)


if __name__ == '__main__':
    main()
