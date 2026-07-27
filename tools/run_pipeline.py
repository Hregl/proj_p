"""Four-stage ground-coordinate pipeline orchestrator.

Stages:
  1. Camera intrinsic calibration (SLS circle grid)
  2. Ground G-frame calibration (C_T_G per image from ground markers)
  3. Aircraft marker 3D triangulation + B-frame construction
  4. Aircraft pose estimation + repeatability evaluation

Each stage explicitly declares its inputs, output directory, and
failure stop conditions. All thresholds are read from the experiment
config YAML.

Usage:
    python tools/run_pipeline.py configs/experiment_ground_pipeline.yaml

Or run individual stages:
    python tools/run_pipeline.py configs/experiment_ground_pipeline.yaml --stage 2
"""

import sys, os, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_stage_01(config: dict, session_dir: Path) -> bool:
    """Stage 1: Camera intrinsic calibration."""
    print("\n" + "=" * 70)
    print("STAGE 1: Camera Intrinsic Calibration")
    print("=" * 70)

    s1 = config.get('stage_01', {})
    if not s1.get('enabled', True):
        print("  Stage 1 disabled, skipping.")
        return True

    images = s1.get('calibration_images', [])
    if not images:
        print("  ERROR: No calibration_images specified in config.")
        print("  Set stage_01.calibration_images to a list of image paths.")
        return False

    # Expand globs
    import glob as globmod
    expanded = []
    for pat in images:
        matches = globmod.glob(pat)
        if matches:
            expanded.extend(matches)
        else:
            expanded.append(pat)

    if not expanded:
        print("  ERROR: No calibration images found.")
        return False

    cmd = [
        sys.executable, 'tools/run_calibration.py',
        *expanded,
        '--circle-interval', str(s1.get('circle_interval_mm', 25)),
        '--output', str(session_dir),
    ]
    if s1.get('smooth', False):
        cmd.append('--smooth')

    print(f"  Running: {' '.join(cmd)}")
    ret = os.system(' '.join(cmd))
    if ret != 0:
        print(f"  FAILED (exit code {ret})")
        return False

    # Verify output
    npz_path = session_dir / 'stage_01_intrinsics.npz'
    if not npz_path.exists():
        print(f"  FAILED: Missing output {npz_path}")
        return False

    print(f"  SUCCESS: {npz_path}")
    return True


def run_stage_02(config: dict, session_dir: Path) -> bool:
    """Stage 2: Ground G-frame calibration."""
    print("\n" + "=" * 70)
    print("STAGE 2: Ground G-frame Calibration (C_T_G per image)")
    print("=" * 70)

    s2 = config.get('stage_02', {})
    if not s2.get('enabled', True):
        print("  Stage 2 disabled, skipping.")
        return True

    images = s2.get('ground_view_images', [])
    if not images:
        print("  ERROR: No ground_view_images specified.")
        return False

    import glob as globmod
    expanded = []
    for pat in images:
        matches = globmod.glob(pat)
        if matches:
            expanded.extend(matches)
        else:
            expanded.append(pat)

    if not expanded:
        print("  ERROR: No ground view images found.")
        return False

    camera_config = config.get('stage_01', {}).get('camera_config',
                                                   'configs/cameras/camera_25mm.yaml')

    stage2_dir = session_dir / 'stage_02'
    cmd = [
        sys.executable, 'tools/estimate_ground_pose.py',
        '--config', camera_config,
        '--ground-3d', s2.get('ground_markers_3d', 'configs/ground_markers_G.yaml'),
        '--annotations', s2.get('ground_annotation_dir', 'annotations/ground_2d'),
        '--output-dir', str(stage2_dir),
        '--ransac-threshold', str(s2.get('ransac_threshold_px', 2.0)),
        '--min-inliers', str(s2.get('min_inliers', 4)),
    ]
    for img in expanded:
        cmd.extend(['--image', img])

    print(f"  Running ground pose estimation on {len(expanded)} images...")
    ret = os.system(' '.join(cmd))
    if ret != 0:
        print(f"  FAILED (exit code {ret})")
        return False

    poses_json = stage2_dir / 'stage_02_ground_poses.json'
    if not poses_json.exists():
        print(f"  FAILED: Missing output {poses_json}")
        return False

    print(f"  SUCCESS: {poses_json}")
    return True


def run_stage_03(config: dict, session_dir: Path) -> bool:
    """Stage 3: Aircraft marker 3D triangulation + B-frame construction."""
    print("\n" + "=" * 70)
    print("STAGE 3: Aircraft Marker 3D Triangulation + B-Frame Construction")
    print("=" * 70)

    s3 = config.get('stage_03', {})
    if not s3.get('enabled', True):
        print("  Stage 3 disabled, skipping.")
        return True

    images = s3.get('triangulation_images', [])
    if not images:
        print("  ERROR: No triangulation_images specified.")
        return False

    import glob as globmod
    expanded = []
    for pat in images:
        matches = globmod.glob(pat)
        if matches:
            expanded.extend(matches)
        else:
            expanded.append(pat)

    if not expanded:
        print("  ERROR: No triangulation images found.")
        return False

    camera_config = config.get('stage_01', {}).get('camera_config',
                                                   'configs/cameras/camera_25mm.yaml')
    stage2_dir = session_dir / 'stage_02'
    ground_poses = stage2_dir / 'stage_02_ground_poses.json'
    if not ground_poses.exists():
        print(f"  ERROR: Stage 2 output not found: {ground_poses}")
        return False

    points_G = session_dir / 'stage_03_aircraft_points_G.yaml'

    # 3a: Triangulation
    cmd_tri = [
        sys.executable, 'tools/triangulate_aircraft_points.py',
        *expanded,
        '--config', camera_config,
        '--ground-poses', str(ground_poses),
        '--output', str(points_G),
        '--max-error', str(s3.get('max_reprojection_error_px', 10.0)),
    ]
    point_names = s3.get('aircraft_point_names', [])
    if point_names:
        cmd_tri.extend(['--point-names'] + point_names)

    print(f"  Running triangulation on {len(expanded)} images...")
    print(f"  (Interactive GUI will open for point labeling)")
    ret = os.system(' '.join(cmd_tri))
    if ret != 0:
        print(f"  WARNING: Triangulation returned exit code {ret}")

    if not points_G.exists():
        print(f"  FAILED: Missing output {points_G}")
        return False

    print(f"  Triangulation: {points_G}")

    # 3b: B-frame construction
    points_B = session_dir / 'stage_03_aircraft_points_B.yaml'
    report_B = session_dir / 'stage_03_B_frame_report.json'
    cmd_b = [
        sys.executable, 'tools/convert_to_B_frame.py',
        '--input', str(points_G),
        '--output', str(points_B),
        '--report', str(report_B),
    ]
    print(f"  Building B-frame...")
    ret = os.system(' '.join(cmd_b))
    if ret != 0 or not points_B.exists():
        print(f"  FAILED: B-frame construction failed")
        return False

    print(f"  B-frame: {points_B}")
    print(f"  Report:  {report_B}")
    print(f"  SUCCESS")
    return True


def run_stage_04(config: dict, session_dir: Path) -> bool:
    """Stage 4: Aircraft pose estimation + repeatability."""
    print("\n" + "=" * 70)
    print("STAGE 4: Aircraft Pose Estimation")
    print("=" * 70)

    s4 = config.get('stage_04', {})
    if not s4.get('enabled', True):
        print("  Stage 4 disabled, skipping.")
        return True

    images = s4.get('pose_images', [])
    if not images:
        print("  ERROR: No pose_images specified.")
        return False

    import glob as globmod
    expanded = []
    for pat in images:
        matches = globmod.glob(pat)
        if matches:
            expanded.extend(matches)
        else:
            expanded.append(pat)

    if not expanded:
        print("  ERROR: No pose images found.")
        return False

    camera_config = config.get('stage_01', {}).get('camera_config',
                                                   'configs/cameras/camera_25mm.yaml')
    stage2_dir = session_dir / 'stage_02'
    ground_poses_json = stage2_dir / 'stage_02_ground_poses.json'
    aircraft_3d_B = s4.get('aircraft_3d_B', 'configs/aircraft_points_B.yaml')
    ann_dir = s4.get('annotation_dir', 'annotations/aircraft_2d')

    stage4_dir = session_dir / 'stage_04'
    stage4_dir.mkdir(parents=True, exist_ok=True)

    # Process each image
    final_csvs = []
    for img_path in expanded:
        stem = Path(img_path).stem
        ann_yaml = Path(ann_dir) / f"{stem}_points.yaml"
        if not ann_yaml.exists():
            print(f"  SKIP {stem}: no annotation at {ann_yaml}")
            continue

        # Ground pose CSV (need to re-derive from ground_poses.json)
        # For simplicity, we read the ground pose from JSON and convert to CSV
        import json as _json
        with open(ground_poses_json, encoding='utf-8') as f:
            gp = _json.load(f)
        if stem not in gp.get('poses', {}):
            print(f"  SKIP {stem}: no ground pose")
            continue

        gp_data = gp['poses'][stem]
        C_R_G = gp_data['C_R_G']
        C_t_G = gp_data['C_t_G']

        import cv2, numpy as np
        rvec_g, _ = cv2.Rodrigues(np.array(C_R_G, dtype=np.float64))

        # Write ground pose CSV
        ground_csv = stage4_dir / f"{stem}_ground_pose.csv"
        with open(ground_csv, 'w', encoding='utf-8') as f:
            f.write('image_id,rvec_x,rvec_y,rvec_z,tvec_x,tvec_y,tvec_z,'
                    'rmse_px,inlier_count\n')
            f.write(f'{stem},{rvec_g[0][0]:.6f},{rvec_g[1][0]:.6f},'
                    f'{rvec_g[2][0]:.6f},'
                    f'{C_t_G[0]:.4f},{C_t_G[1]:.4f},{C_t_G[2]:.4f},'
                    f'{gp_data["rmse_px"]:.4f},{gp_data["n_inliers"]}\n')

        # Aircraft PnP
        ac_csv = stage4_dir / f"{stem}_aircraft_pose.csv"
        cmd_ac = [
            sys.executable, 'tools/estimate_aircraft_pose.py',
            '--config', camera_config,
            '--aircraft-3d', aircraft_3d_B,
            '--aircraft-2d', str(ann_yaml),
            '--output', str(ac_csv),
        ]
        ret = os.system(' '.join(cmd_ac))
        if ret != 0:
            print(f"  FAIL {stem}: aircraft PnP failed")
            continue

        # Compose final pose
        final_csv = stage4_dir / f"{stem}_final_pose.csv"
        cmd_comp = [
            sys.executable, 'tools/compose_aircraft_pose.py',
            '--config', camera_config,
            '--ground-pose', str(ground_csv),
            '--aircraft-pose', str(ac_csv),
            '--output', str(final_csv),
        ]
        ret = os.system(' '.join(cmd_comp))
        if ret != 0:
            print(f"  FAIL {stem}: pose composition failed")
            continue

        final_csvs.append(str(final_csv))
        print(f"  OK {stem} -> {final_csv}")

    if not final_csvs:
        print("  FAILED: No successful poses")
        return False

    # Repeatability
    repeat_cfg = config.get('repeatability', {})
    rep_json = stage4_dir / 'stage_04_repeatability_report.json'
    cmd_rep = [
        sys.executable, 'tools/evaluate_repeatability.py',
        *final_csvs,
        '--group-name', config.get('session', 'experiment'),
        '--output', str(rep_json),
        '--max-ground-rmse',
        str(repeat_cfg.get('max_ground_rmse_px', 3.0)),
        '--max-aircraft-rmse',
        str(repeat_cfg.get('max_aircraft_rmse_px', 5.0)),
    ]
    os.system(' '.join(cmd_rep))

    print(f"\n  Final poses: {len(final_csvs)} successful")
    print(f"  Repeatability: {rep_json}")
    print(f"  SUCCESS")
    return True


def main():
    import argparse, yaml
    parser = argparse.ArgumentParser(
        description='Four-stage ground-coordinate pipeline orchestrator')
    parser.add_argument('config', help='Experiment config YAML')
    parser.add_argument('--stage', type=int, choices=[1, 2, 3, 4],
                       help='Run a single stage (1-4) instead of all')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print stage commands without executing')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    session_dir = Path(config.get('output_dir', 'output/experiment'))
    session_dir.mkdir(parents=True, exist_ok=True)

    stages = {
        1: ('Camera Intrinsics', lambda: run_stage_01(config, session_dir)),
        2: ('Ground G-frame', lambda: run_stage_02(config, session_dir)),
        3: ('Aircraft Triangulation', lambda: run_stage_03(config, session_dir)),
        4: ('Pose Estimation', lambda: run_stage_04(config, session_dir)),
    }

    if args.stage:
        name, fn = stages[args.stage]
        success = fn()
        if not success:
            print(f"\nPipeline stopped at Stage {args.stage} ({name}).")
            sys.exit(1)
        print(f"\nStage {args.stage} ({name}) complete.")
        return

    # Run all stages
    start_time = time.time()
    for stage_num in [1, 2, 3, 4]:
        name, fn = stages[stage_num]
        print(f"\n{'#' * 70}")
        print(f"# Stage {stage_num}/4: {name}")
        print(f"{'#' * 70}")

        success = fn()
        if not success:
            print(f"\n{'=' * 70}")
            print(f"PIPELINE STOPPED at Stage {stage_num} ({name}).")
            print(f"Fix the failure and re-run from this stage:")
            print(f"  python tools/run_pipeline.py {args.config} --stage {stage_num}")
            print(f"{'=' * 70}")
            sys.exit(1)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"PIPELINE COMPLETE ({elapsed:.0f}s)")
    print(f"Output: {session_dir}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
