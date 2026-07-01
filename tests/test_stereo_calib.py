"""合成双目标定精度测试。"""
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
    print("双目标定 — 合成精度测试")
    print("=" * 60)

    # 生成数据
    (
        left_imgs, right_imgs,
        left_corners, right_corners,
        obj_pts,
        K_left, dist_left, K_right, dist_right,
        R_gt, T_gt,
    ) = _generate_synthetic_stereo_data(n_pairs=15, noise_px=0.3)

    n = len(left_imgs)
    print(f"\n[1] 生成了 {n} 对图像 ({left_imgs[0].shape[1]}x{left_imgs[0].shape[0]})")

    # 标定
    print("\n[2] 正在运行双目标定 …")
    calib = StereoCalibrator(pattern_type="chessboard",
                             pattern_size=(9, 6), square_size=0.025)

    sp = calib.calibrate_stereo_from_corners(
        left_corners, right_corners, obj_pts,
        K_left, dist_left, K_right, dist_right,
        image_size=(1280, 720), fix_intrinsics=True, debug=True,
    )
    assert sp is not None, "标定失败!"

    # 检查旋转精度
    R_err_mat = sp.R @ R_gt.T
    angle_err = math.acos(np.clip((np.trace(R_err_mat) - 1) / 2, -1.0, 1.0))
    print(f"\n[3] 旋转误差: {np.rad2deg(angle_err):.5f} 度")
    assert angle_err < np.deg2rad(0.5), f"旋转误差过高: {np.rad2deg(angle_err):.5f} 度"

    # 检查基线精度
    t_gt_norm = np.linalg.norm(T_gt)
    t_est_norm = np.linalg.norm(sp.T)
    baseline_err_pct = abs(t_est_norm - t_gt_norm) / t_gt_norm * 100
    print(f"  基线误差: {baseline_err_pct:.3f}% "
          f"(真值={t_gt_norm*1000:.1f}, 估计={t_est_norm*1000:.1f} mm)")
    assert baseline_err_pct < 1.0, f"基线误差过高: {baseline_err_pct:.3f}%"

    # 检查重投影
    print(f"  重投影 RMS: {sp.rms_error:.5f} px")
    assert sp.rms_error < 2.0, f"重投影误差过高: {sp.rms_error:.5f} px"

    print("\n通过: 所有精度阈值均已满足。")
    print(calib.summary())


if __name__ == "__main__":
    main()
