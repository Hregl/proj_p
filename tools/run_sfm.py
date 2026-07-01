"""多视图 SfM 重建命令行工具。

用法:
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

    # 加载相机参数
    calib = np.load(args.camera)
    K = calib["camera_matrix"]
    dist = calib["dist_coeffs"].ravel()

    # 加载图像
    images = []
    for p in args.images:
        img = cv2.imread(p)
        if img is None:
            print(f"警告: 无法读取 '{p}', 跳过")
            continue
        images.append(img)

    if len(images) < 2:
        print("错误: SfM 至少需要 2 张图像")
        sys.exit(1)

    print(f"已加载 {len(images)} 张图像")

    # 运行 SfM
    sfm = MultiViewSfM(K, dist, marker_size_m=args.marker_size,
                        aruco_dict=args.aruco_dict)
    sfm.add_views(images)

    if not sfm.initialize():
        print("错误: SfM 初始化失败")
        sys.exit(1)

    n_new = sfm.register_all()
    n_reg = sum(1 for v in sfm.views if v.registered)
    print(f"已注册: {n_reg}/{len(sfm.views)} 个视图")

    err = sfm.bundle_adjust(iterations=5, use_sparse_lm=True, verbose=True)

    # 对齐到地面
    if args.ground_ids:
        sfm.align_to_ground(args.ground_ids)

    print(sfm.summary())

    # 保存
    pts = sfm.points_3d
    ids_sorted = sorted(pts.keys())
    pts_arr = np.array([pts[i] for i in ids_sorted], dtype=np.float64)
    np.savez(args.output,
             marker_ids=np.array(ids_sorted),
             points_3d=pts_arr,
             reproj_error=err)
    print(f"\n已保存到 {args.output}")


if __name__ == "__main__":
    main()
