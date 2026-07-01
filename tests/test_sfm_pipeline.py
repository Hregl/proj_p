"""合成 SfM 精度测试 — 验证重建精度。"""
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
    print("SfM 流水线 — 合成精度测试")
    print("=" * 60)

    # 生成场景
    print("\n[1] 正在生成合成场景 …")
    images, K, dist, gt_pts, gt_poses = _generate_synthetic_scene(
        n_markers=15, n_views=6, noise_px=0.0,
    )

    sfm = MultiViewSfM(K, dist, aruco_dict="4x4_50")
    for vi, img in enumerate(images):
        sfm.views.append(View(f"view_{vi}", img))
    _inject_projections(sfm, gt_pts, gt_poses, noise_px=0.5)

    # 运行流水线
    print("\n[2] 正在运行 SfM …")
    assert sfm.initialize(), "初始化失败!"
    n_new = sfm.register_all()
    assert n_new >= 2, f"仅注册了 {n_new} 个视图"
    err = sfm.bundle_adjust(iterations=5, use_sparse_lm=True, verbose=False)

    # 使用真值测量进行对齐
    ground_ids = list(range(8))
    ground_gt = [gt_pts[i] for i in ground_ids]
    sfm.align_to_ground(ground_ids, ground_plane_points=ground_gt)

    # 检查 3D 点精度
    errors = []
    for mid in gt_pts:
        if mid in sfm._points3d:
            gt = np.array(gt_pts[mid])
            rec = np.array(sfm._points3d[mid])
            errors.append(np.linalg.norm(gt - rec) * 1000)

    mean_err = np.mean(errors)
    max_err = np.max(errors)
    print(f"\n[3] 精度: 平均={mean_err:.2f} mm, 最大={max_err:.2f} mm")

    # 检查相机姿态精度
    rot_errors = []
    for vi, view in enumerate(sfm.views):
        if not view.registered:
            continue
        gt_R, gt_t = gt_poses[vi]
        R_diff = view.R @ gt_R.T
        angle_err = math.acos(
            np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0))
        rot_errors.append(np.rad2deg(angle_err))

    print(f"  相机旋转误差: 平均={np.mean(rot_errors):.3f} 度")

    # 通过/失败阈值
    assert mean_err < 1.0, f"3D 误差过高: {mean_err:.2f} mm"
    assert max_err < 2.0, f"最大 3D 误差过高: {max_err:.2f} mm"
    assert np.mean(rot_errors) < 0.5, f"旋转误差过高: {np.mean(rot_errors):.3f} 度"

    print("\n通过: 所有精度阈值均已满足。")
    print(sfm.summary())


if __name__ == "__main__":
    main()
