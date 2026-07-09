"""
P1: Reprojection error analysis module.

Computes per-point and per-view reprojection errors after PnP.
Identifies systematic outliers and generates summary statistics.

Usage:
  python tools/analyze_reprojection.py \
      --aircraft-3d configs/aircraft_points.yaml \
      --aircraft-2d annotations/aircraft_2d/MVIMG_20260707_202357_points.yaml \
      --camera-pose output/MVIMG_20260707_202357_board_pose.csv
"""
import sys, yaml, csv, cv2, numpy as np, math
from pathlib import Path

def load_csv_pose(filepath):
    """Load rvec/tvec from board/aircraft PnP CSV."""
    with open(filepath) as f:
        for line in f:
            if line.startswith('image_id'): continue
            parts = line.strip().split(',')
            if len(parts) >= 7:
                rvec = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                tvec = np.array([float(parts[4]), float(parts[5]), float(parts[6])])
                R, _ = cv2.Rodrigues(rvec)
                return R, tvec.ravel(), rvec, float(parts[7]) if len(parts) > 7 else 0
    return None, None, None, 0

def main():
    import argparse
    p = argparse.ArgumentParser(
        description='Analyze per-point reprojection errors after PnP')
    p.add_argument('--config', default='configs/experiment_config.yaml')
    p.add_argument('--aircraft-3d', default='configs/aircraft_points.yaml')
    p.add_argument('--aircraft-2d', required=True, help='Aircraft 2D annotation YAML')
    p.add_argument('--camera-pose', required=True,
                   help='Camera pose CSV (from estimate_board_pose.py)')
    p.add_argument('--output', '-o', default=None, help='Output CSV for per-point errors')
    args = p.parse_args()

    # Load camera
    with open(args.config, encoding='utf-8') as f: exp = yaml.safe_load(f)
    cal = exp['calibration']
    K = np.array([[cal['fx'], 0, cal['cx']],
                   [0, cal['fy'], cal['cy']],
                   [0, 0, 1]], dtype=np.float64)
    dist = np.array(cal['dist'], dtype=np.float64)

    # Load 3D points
    with open(args.aircraft_3d, encoding='utf-8') as f: ac3d = yaml.safe_load(f)

    # Load 2D annotations
    with open(args.aircraft_2d, encoding='utf-8') as f: ac2d = yaml.safe_load(f)

    # Load camera pose (C_T_G: board → camera)
    C_R_G, C_t_G, rvec, board_rmse = load_csv_pose(args.camera_pose)
    if C_R_G is None:
        print("Invalid camera pose CSV"); sys.exit(1)

    # Build 3D-2D correspondences
    obj, img, names = [], [], []
    pts3d_all = ac3d.get('points', {})
    if ac3d.get('points_chinese'):
        pts3d_all = {**pts3d_all, **ac3d['points_chinese']}
    for name, info in pts3d_all.items():
        if name in ac2d.get('points', {}):
            p2d = ac2d['points'][name]
            px, py = float(p2d['pixel_x']), float(p2d['pixel_y'])
            if px < 0 or py < 0:
                continue
            x, y, z = float(info['x_mm']), float(info['y_mm']), float(info['z_mm'])
            # Transform 3D point from G frame to C frame
            pG = np.array([x, y, z])
            pC = C_R_G @ pG + C_t_G
            obj.append(pC)
            img.append([px, py])
            names.append(name)

    if len(obj) < 4:
        print(f'Insufficient points: {len(obj)}'); sys.exit(1)

    obj_arr = np.array(obj, dtype=np.float64)
    img_arr = np.array(img, dtype=np.float64)

    # PnP: aircraft in camera frame
    success, rvec_ac, tvec_ac, inliers = cv2.solvePnPRansac(
        obj_arr, img_arr, K, dist,
        flags=cv2.SOLVEPNP_EPNP, iterationsCount=200,
        reprojectionError=3.0, confidence=0.99)

    if not success:
        print('Aircraft PnP failed'); sys.exit(1)

    # Project ALL 3D points using the PnP solution
    proj_all, _ = cv2.projectPoints(obj_arr, rvec_ac, tvec_ac, K, dist)
    proj_all = proj_all.reshape(-1, 2)

    # Per-point errors
    errors = []
    for i, name in enumerate(names):
        err = float(np.linalg.norm(proj_all[i] - img_arr[i]))
        errors.append((name, err, img_arr[i][0], img_arr[i][1],
                       proj_all[i][0], proj_all[i][1]))

    errors.sort(key=lambda x: -x[1])  # sort by error desc
    rmse = math.sqrt(np.mean([e[1]**2 for e in errors]))
    max_err = errors[0][1]
    min_err = errors[-1][1]

    inlier_set = set(inliers.ravel()) if inliers is not None else set()

    print(f"\n{'='*60}")
    print(f"Reprojection Error Analysis")
    print(f"{'='*60}")
    print(f"  Image: {Path(args.aircraft_2d).stem}")
    print(f"  Points: {len(errors)}, RMSE: {rmse:.3f}px")
    print(f"  Max error: {max_err:.1f}px, Min error: {min_err:.1f}px")
    print(f"  Board RMSE: {board_rmse:.3f}px")

    print(f"\n  {'Point':<14} {'Error':>8} {'Observed':>16} {'Projected':>16} {'Status'}")
    print(f"  {'':-<14} {'':->8} {'':->16} {'':->16} {'':-<8}")
    for name, err, ox, oy, px, py in errors:
        status = "INLIER" if errors.index((name, err, ox, oy, px, py)) == -1 or \
                  any(i < len(names) and names[i] == name and i in inlier_set
                      for i in range(len(names))) else "OUTLIER"
        # Find index in names
        idx = names.index(name)
        is_inlier = idx in inlier_set
        status = "OK" if is_inlier else "OUTLIER"
        flag = " *** OUTLIER" if not is_inlier else ""
        print(f"  {name:<14} {err:>7.2f}px  ({ox:>6.0f},{oy:>6.0f}) -> "
              f"({px:>6.0f},{py:>6.0f}){flag}")

    n_inl = len(inlier_set)
    print(f"\n  Inliers: {n_inl}/{len(errors)}")

    # Per-axis error statistics
    dx = [e[2] - e[4] for e in errors]  # observed_x - projected_x
    dy = [e[3] - e[5] for e in errors]  # observed_y - projected_y
    print(f"\n  Error breakdown:")
    print(f"    X (u): mean={np.mean(dx):.2f}px, std={np.std(dx):.2f}px")
    print(f"    Y (v): mean={np.mean(dy):.2f}px, std={np.std(dy):.2f}px")

    if args.output:
        with open(args.output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image','point','error_px','obs_u','obs_v',
                            'proj_u','proj_v','inlier','board_rmse_px','rmse_px'])
            for name, err, ox, oy, px, py in errors:
                idx = names.index(name)
                writer.writerow([Path(args.aircraft_2d).stem, name,
                                round(err, 3), round(ox, 1), round(oy, 1),
                                round(px, 1), round(py, 1),
                                1 if idx in inlier_set else 0,
                                round(board_rmse, 3), round(rmse, 3)])
        print(f"  -> {args.output}")

if __name__ == '__main__':
    main()
