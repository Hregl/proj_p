"""Multi-view SfM reconstruction CLI.

Usage:
    python tools/run_sfm.py data/sfm/*.png --camera camera.npz \
        --ground-ids 0 1 2 3 --aircraft-ids 4 5 6
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2
import numpy as np
from sls_calib import MultiViewSfM


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Multi-view SfM from ArUco marker images")
    parser.add_argument("images", nargs="+", help="Input images (multiple views)")
    parser.add_argument("--camera", "-c", required=True,
                        help="Camera intrinsics .npz file")
    parser.add_argument("--marker-size", type=float, default=None,
                        help="ArUco marker physical size (metres)")
    parser.add_argument("--aruco-dict", default="4x4_50",
                        help="ArUco dictionary (default: 4x4_50)")
    parser.add_argument("--ground-ids", type=int, nargs="+", default=[],
                        help="ArUco IDs of ground markers")
    parser.add_argument("--output", "-o", default="sfm_result.npz",
                        help="Output file (default: sfm_result.npz)")
    args = parser.parse_args()

    # Load camera
    calib = np.load(args.camera)
    K = calib["camera_matrix"]
    dist = calib["dist_coeffs"].ravel()

    # Load images
    images = []
    for p in args.images:
        img = cv2.imread(p)
        if img is None:
            print(f"Warning: cannot read '{p}', skipping")
            continue
        images.append(img)

    if len(images) < 2:
        print("Error: need at least 2 images for SfM")
        sys.exit(1)

    print(f"Loaded {len(images)} images")

    # Run SfM
    sfm = MultiViewSfM(K, dist, marker_size_m=args.marker_size,
                        aruco_dict=args.aruco_dict)
    sfm.add_views(images)

    if not sfm.initialize():
        print("Error: SfM initialisation failed")
        sys.exit(1)

    n_new = sfm.register_all()
    n_reg = sum(1 for v in sfm.views if v.registered)
    print(f"Registered: {n_reg}/{len(sfm.views)} views")

    err = sfm.bundle_adjust(iterations=5, use_sparse_lm=True, verbose=True)

    # Align to ground
    if args.ground_ids:
        sfm.align_to_ground(args.ground_ids)

    print(sfm.summary())

    # Save
    pts = sfm.points_3d
    ids_sorted = sorted(pts.keys())
    pts_arr = np.array([pts[i] for i in ids_sorted], dtype=np.float64)
    np.savez(args.output,
             marker_ids=np.array(ids_sorted),
             points_3d=pts_arr,
             reproj_error=err)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
