"""Synthetic SfM accuracy test — verifies reconstruction accuracy."""
import math
import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sls_calib import MultiViewSfM, View
from sls_calib.sfm_pipeline import _generate_synthetic_scene, _inject_projections


def main():
    print("=" * 60)
    print("SfM Pipeline — Synthetic Accuracy Test")
    print("=" * 60)

    # Generate scene
    print("\n[1] Generating synthetic scene …")
    images, K, dist, gt_pts, gt_poses = _generate_synthetic_scene(
        n_markers=15, n_views=6, noise_px=0.0,
    )

    sfm = MultiViewSfM(K, dist, aruco_dict="4x4_50")
    for vi, img in enumerate(images):
        sfm.views.append(View(f"view_{vi}", img))
    _inject_projections(sfm, gt_pts, gt_poses, noise_px=0.5)

    # Run pipeline
    print("\n[2] Running SfM …")
    assert sfm.initialize(), "Init failed!"
    n_new = sfm.register_all()
    assert n_new >= 2, f"Only {n_new} views registered"
    err = sfm.bundle_adjust(iterations=5, use_sparse_lm=True, verbose=False)

    # Align to ground with GT measurements
    ground_ids = list(range(8))
    ground_gt = [gt_pts[i] for i in ground_ids]
    sfm.align_to_ground(ground_ids, ground_plane_points=ground_gt)

    # Check 3D point accuracy
    errors = []
    for mid in gt_pts:
        if mid in sfm._points3d:
            gt = np.array(gt_pts[mid])
            rec = np.array(sfm._points3d[mid])
            errors.append(np.linalg.norm(gt - rec) * 1000)

    mean_err = np.mean(errors)
    max_err = np.max(errors)
    print(f"\n[3] Accuracy: mean={mean_err:.2f} mm, max={max_err:.2f} mm")

    # Check camera pose accuracy
    rot_errors = []
    for vi, view in enumerate(sfm.views):
        if not view.registered:
            continue
        gt_R, gt_t = gt_poses[vi]
        R_diff = view.R @ gt_R.T
        angle_err = math.acos(
            np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0))
        rot_errors.append(np.rad2deg(angle_err))

    print(f"  Camera rotation error: mean={np.mean(rot_errors):.3f} deg")

    # Pass/fail thresholds
    assert mean_err < 1.0, f"3D error too high: {mean_err:.2f} mm"
    assert max_err < 2.0, f"Max 3D error too high: {max_err:.2f} mm"
    assert np.mean(rot_errors) < 0.5, f"Rotation error too high: {np.mean(rot_errors):.3f} deg"

    print("\nPASS: All accuracy thresholds met.")
    print(sfm.summary())


if __name__ == "__main__":
    main()
