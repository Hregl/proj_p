"""双目标定命令行工具。

用法:
    python tools/run_stereo_calib.py --left left_*.png --right right_*.png \
        --pattern chessboard --pattern-size 9 6 --square-size 0.025
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2
import numpy as np
from sls_calib import StereoCalibrator


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Stereo camera calibration")
    parser.add_argument("--left", nargs="+", required=True,
                        help="Left camera images")
    parser.add_argument("--right", nargs="+", required=True,
                        help="Right camera images")
    parser.add_argument("--pattern", default="chessboard",
                        choices=["chessboard", "circles", "charuco"],
                        help="Calibration pattern type (default: chessboard)")
    parser.add_argument("--pattern-size", type=int, nargs=2, default=[9, 6],
                        help="Inner corners: cols rows (default: 9 6)")
    parser.add_argument("--square-size", type=float, default=0.025,
                        help="Square side length in metres (default: 0.025)")
    parser.add_argument("--aruco-dict", default="4x4_50",
                        help="ArUco dictionary for ChArUco (default: 4x4_50)")
    parser.add_argument("--output", "-o", default="stereo_params.npz",
                        help="Output file (default: stereo_params.npz)")
    args = parser.parse_args()

    if len(args.left) != len(args.right):
        print(f"错误: {len(args.left)} 张左图 != {len(args.right)} 张右图")
        sys.exit(1)

    left_imgs = []
    right_imgs = []
    for pl, pr in zip(args.left, args.right):
        imgL = cv2.imread(pl)
        imgR = cv2.imread(pr)
        if imgL is None or imgR is None:
            print(f"警告: 跳过图像对 ({pl}, {pr})")
            continue
        left_imgs.append(imgL)
        right_imgs.append(imgR)

    n = len(left_imgs)
    if n < 5:
        print(f"错误: 需要至少 5 对有效图像, 实际仅 {n} 对")
        sys.exit(1)

    print(f"已加载 {n} 对立体图像")

    calib = StereoCalibrator(
        pattern_type=args.pattern,
        pattern_size=tuple(args.pattern_size),
        square_size=args.square_size,
        aruco_dict_name=args.aruco_dict,
    )

    # 内参标定
    K_l, d_l, rms_l = calib.calibrate_intrinsics(left_imgs, debug=True)
    K_r, d_r, rms_r = calib.calibrate_intrinsics(right_imgs, debug=True)

    if K_l is None or K_r is None:
        print("错误: 内参标定失败")
        sys.exit(1)

    # 双目标定
    sp = calib.calibrate_stereo(left_imgs, right_imgs,
                                K_l, d_l, K_r, d_r, debug=True)
    if sp is None:
        print("错误: 双目标定失败")
        sys.exit(1)

    print(calib.summary())

    np.savez(args.output,
             K_left=sp.K_left, dist_left=sp.dist_left,
             K_right=sp.K_right, dist_right=sp.dist_right,
             R=sp.R, T=sp.T, E=sp.E, F=sp.F,
             R1=sp.R1, R2=sp.R2, P1=sp.P1, P2=sp.P2, Q=sp.Q)
    print(f"\n已保存到 {args.output}")


if __name__ == "__main__":
    main()
