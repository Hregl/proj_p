"""Stage 2: Ground coordinate system (G) calibration.

Estimates C_T_G (camera-to-ground pose) for each image using
physically measured ground marker 3D coordinates and their 2D
detections. No longer relies on calibration board for world frame.

Usage:
    # Single image:
    python tools/estimate_ground_pose.py --config configs/cameras/camera_25mm.yaml \\
        --ground-3d configs/ground_markers_G.yaml \\
        --image data/ground_views/img_001.png \\
        --annotations annotations/ground_2d/

    # Batch:
    python tools/estimate_ground_pose.py --config configs/cameras/camera_25mm.yaml \\
        --ground-3d configs/ground_markers_G.yaml \\
        --images data/ground_views/*.png \\
        --annotations annotations/ground_2d/ \\
        --output-dir output/session_01/
"""

import sys, yaml, cv2, numpy as np, json, math, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sls_calib.ground_pose import GroundPoseEstimator, GroundPoseResult
from sls_calib.config_validator import load_camera_config


def load_annotations_yaml(yaml_path: str) -> Dict[str, Tuple[float, float]]:
    """Load 2D point annotations from a per-image YAML file.

    Returns {point_id: (u, v)} for visible points only.
    """
    with open(yaml_path, encoding='utf-8') as f:
        data = yaml.safe_load(f)

    pts = {}
    for name, info in data.get('points', {}).items():
        px = float(info.get('pixel_x', -1))
        py = float(info.get('pixel_y', -1))
        visible = info.get('visible', True)
        if px >= 0 and py >= 0 and visible:
            pts[name] = (px, py)
    return pts


def draw_ground_axes(image: np.ndarray, result: GroundPoseResult,
                     K: np.ndarray, dist: np.ndarray,
                     axis_length: float = 200.0) -> np.ndarray:
    """Draw G-frame axes and ground marker projections for visual check."""
    if not result.success:
        return image

    out = image.copy()
    rvec = result.rvec
    tvec = result.tvec

    # Draw G axes at origin
    axis_pts = np.array([[0, 0, 0],
                         [axis_length, 0, 0],
                         [0, axis_length, 0],
                         [0, 0, axis_length]], dtype=np.float32)
    proj, _ = cv2.projectPoints(axis_pts, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2).astype(int)
    origin = tuple(proj[0])

    axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # X=red, Y=green, Z=blue
    axis_labels = ['X_G', 'Y_G', 'Z_G']
    for i in range(3):
        cv2.line(out, origin, tuple(proj[i + 1]), axis_colors[i], 2)
        cv2.putText(out, axis_labels[i],
                    (proj[i + 1][0] + 5, proj[i + 1][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, axis_colors[i], 1)
    cv2.circle(out, origin, 4, (0, 255, 255), -1)
    cv2.putText(out, 'G', (origin[0] + 8, origin[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Project ground marker 3D positions
    est = None  # will be set if we have the estimator
    return out


def main():
    import argparse, glob as globmod

    p = argparse.ArgumentParser(
        description='Stage 2: Ground coordinate system calibration (C_T_G per image)')
    p.add_argument('--config', required=True,
                   help='Camera config YAML (e.g. configs/cameras/camera_25mm.yaml)')
    p.add_argument('--ground-3d', default='configs/ground_markers_G.yaml',
                   help='Ground marker 3D coordinates YAML')
    p.add_argument('--images', nargs='+',
                   help='Image file paths (supports glob patterns)')
    p.add_argument('--image', help='Single image (alternative to --images)')
    p.add_argument('--annotations', default='annotations/ground_2d',
                   help='Directory with per-image 2D annotation YAML files')
    p.add_argument('--output-dir', '-o', default='output/ground_pose',
                   help='Output directory for pose results')
    p.add_argument('--ransac-threshold', type=float, default=2.0,
                   help='RANSAC reprojection error threshold (px)')
    p.add_argument('--min-inliers', type=int, default=4,
                   help='Minimum RANSAC inliers for success')
    p.add_argument('--draw-axes', action='store_true',
                   help='Export images with G-frame axes overlaid')
    args = p.parse_args()

    # --- Load camera ---
    _, K, dist = load_camera_config(args.config)

    # --- Collect images ---
    if args.images:
        image_paths = list(args.images)
    elif args.image:
        image_paths = [args.image]
    else:
        print("Error: specify --images or --image"); sys.exit(1)

    # Expand globs
    expanded = []
    for pat in image_paths:
        matches = globmod.glob(pat)
        if matches:
            expanded.extend(matches)
        else:
            expanded.append(pat)
    image_paths = expanded

    print(f"Processing {len(image_paths)} images...")
    print(f"  Ground markers: {args.ground_3d}")
    print(f"  Camera config:  {args.config}")
    print(f"  Output dir:     {args.output_dir}")

    # --- Initialize estimator ---
    estimator = GroundPoseEstimator(
        K, dist,
        ground_markers_yaml=args.ground_3d,
        ransac_threshold_px=args.ransac_threshold,
        min_inliers=args.min_inliers,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Process each image ---
    results: List[GroundPoseResult] = []
    n_success, n_fail = 0, 0

    for img_path in image_paths:
        stem = Path(img_path).stem
        ann_path = Path(args.annotations) / f"{stem}_points.yaml"

        if not ann_path.exists():
            print(f"  [{stem}] SKIP: no annotation file at {ann_path}")
            n_fail += 1
            continue

        marker_2d = load_annotations_yaml(str(ann_path))
        result = estimator.estimate(stem, marker_2d, detection_count=len(marker_2d))

        if result.success:
            n_success += 1
            print(f"  [{stem}] OK: {result.n_inliers}/{result.n_matched} inliers, "
                  f"RMSE={result.rmse_px:.2f}px")
        else:
            n_fail += 1
            print(f"  [{stem}] FAIL: {result.failure_reason}")

        for w in result.warnings:
            print(f"         WARN: {w}")

        results.append(result)

        # --- Draw axes visualization ---
        if args.draw_axes and result.success:
            img = cv2.imread(img_path)
            if img is not None:
                vis = draw_ground_axes(img, result, K, dist)
                vis_path = out_dir / f"{stem}_G_axes.png"
                cv2.imwrite(str(vis_path), vis)

    # --- Generate structured outputs ---
    # (1) Poses JSON (for Stage 3)
    poses_json = out_dir / "stage_02_ground_poses.json"
    poses_data = {
        'coordinate_system': 'G',
        'unit': 'mm',
        'ground_markers_yaml': args.ground_3d,
        'poses': {}
    }
    for r in results:
        if r.success:
            poses_data['poses'][r.image_id] = {
                'C_R_G': r.C_R_G.tolist(),
                'C_t_G': r.C_t_G.tolist(),
                'rmse_px': r.rmse_px,
                'n_inliers': r.n_inliers,
                'inlier_ratio': r.inlier_ratio,
            }

    with open(poses_json, 'w', encoding='utf-8') as f:
        json.dump(poses_data, f, indent=2, ensure_ascii=False)
    print(f"\nPoses saved: {poses_json}")

    # (2) Report JSON
    report_json = out_dir / "stage_02_ground_pose_report.json"
    valid_rmse = [r.rmse_px for r in results if r.success]
    valid_inliers = [r.n_inliers for r in results if r.success]

    report = {
        'session_id': out_dir.name,
        'stage': 'ground_pose',
        'status': 'pass' if n_fail == 0 else ('partial' if n_success > 0 else 'fail'),
        'inputs': {
            'n_images': len(image_paths),
            'camera_config': args.config,
            'ground_markers_yaml': args.ground_3d,
            'n_ground_markers': len(estimator.ground_points_3d),
        },
        'outputs': {
            'valid_frames': n_success,
            'failed_frames': n_fail,
        },
        'metrics': {
            'rmse_px_mean': round(float(np.mean(valid_rmse)), 3) if valid_rmse else None,
            'rmse_px_median': round(float(np.median(valid_rmse)), 3) if valid_rmse else None,
            'rmse_px_max': round(float(np.max(valid_rmse)), 3) if valid_rmse else None,
            'inlier_count_mean': round(float(np.mean(valid_inliers)), 1) if valid_inliers else None,
        },
        'thresholds': {
            'ransac_reprojection_px': args.ransac_threshold,
            'min_inliers': args.min_inliers,
        },
        'warnings': [],
        'failed_items': [r.image_id for r in results if not r.success],
        'per_frame': [r.to_dict() for r in results],
    }

    with open(report_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved: {report_json}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Stage 2 Summary: {n_success}/{len(image_paths)} valid poses, "
          f"{n_fail} failed")
    if valid_rmse:
        print(f"  RMSE: mean={np.mean(valid_rmse):.3f}, "
              f"median={np.median(valid_rmse):.3f}, "
              f"max={np.max(valid_rmse):.3f} px")


if __name__ == '__main__':
    main()
