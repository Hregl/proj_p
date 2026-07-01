"""Synthetic stereo calibration accuracy test."""
import math
import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sls_calib import StereoCalibrator
from sls_calib.stereo_calib import _generate_synthetic_stereo_data


def main():
    print("=" * 60)
    print("Stereo Calibration — Synthetic Accuracy Test")
    print("=" * 60)

    # Generate data
    (
        left_imgs, right_imgs,
        left_corners, right_corners,
        obj_pts,
        K_left, dist_left, K_right, dist_right,
        R_gt, T_gt,
    ) = _generate_synthetic_stereo_data(n_pairs=15, noise_px=0.3)

    n = len(left_imgs)
    print(f"\n[1] Generated {n} pairs ({left_imgs[0].shape[1]}x{left_imgs[0].shape[0]})")

    # Calibrate
    print("\n[2] Running stereo calibration …")
    calib = StereoCalibrator(pattern_type="chessboard",
                             pattern_size=(9, 6), square_size=0.025)

    sp = calib.calibrate_stereo_from_corners(
        left_corners, right_corners, obj_pts,
        K_left, dist_left, K_right, dist_right,
        image_size=(1280, 720), fix_intrinsics=True, debug=True,
    )
    assert sp is not None, "Calibration failed!"

    # Check rotation accuracy
    R_err_mat = sp.R @ R_gt.T
    angle_err = math.acos(np.clip((np.trace(R_err_mat) - 1) / 2, -1.0, 1.0))
    print(f"\n[3] Rotation error: {np.rad2deg(angle_err):.5f} deg")
    assert angle_err < np.deg2rad(0.5), f"Rotation error too high: {np.rad2deg(angle_err):.5f} deg"

    # Check baseline accuracy
    t_gt_norm = np.linalg.norm(T_gt)
    t_est_norm = np.linalg.norm(sp.T)
    baseline_err_pct = abs(t_est_norm - t_gt_norm) / t_gt_norm * 100
    print(f"  Baseline error: {baseline_err_pct:.3f}% "
          f"(GT={t_gt_norm*1000:.1f}, Est={t_est_norm*1000:.1f} mm)")
    assert baseline_err_pct < 1.0, f"Baseline error too high: {baseline_err_pct:.3f}%"

    # Check reprojection
    print(f"  Reprojection RMS: {sp.rms_error:.5f} px")
    assert sp.rms_error < 2.0, f"Reprojection error too high: {sp.rms_error:.5f} px"

    print("\nPASS: All accuracy thresholds met.")
    print(calib.summary())


if __name__ == "__main__":
    main()
