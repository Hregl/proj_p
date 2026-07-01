"""单相机内参标定命令行工具。

用法:
    python tools/run_calibration.py data/calib_001.png --circle-interval 35
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2
import numpy as np
from sls_calib import CalibImage, Calibrator


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Calibrate a camera using an SLS circular-dot target")
    parser.add_argument("images", nargs="+", help="Calibration target images")
    parser.add_argument("--circle-interval", type=float, default=35.0,
                        help="Physical spacing between circles (mm) (default: 35)")
    parser.add_argument("--output", "-o", default="camera.npz",
                        help="Output file for K and dist (default: camera.npz)")
    parser.add_argument("--smooth", action="store_true",
                        help="Apply Gaussian blur before detection")
    args = parser.parse_args()

    # 加载图像
    calib_imgs = []
    for i, p in enumerate(args.images):
        img = cv2.imread(p)
        if img is None:
            print(f"警告: 无法读取 '{p}', 跳过")
            continue
        calib_imgs.append(CalibImage(name=f"calib_{i}", image=img, selected=True))

    if len(calib_imgs) < 1:
        print("错误: 没有加载到有效图像")
        sys.exit(1)

    print(f"已加载 {len(calib_imgs)} 张图像")

    # 检测并分配网格
    calib = Calibrator()
    err = calib.extract_circles(calib_imgs, only_selected=True,
                                 smooth=args.smooth, debug=True)
    if err:
        print(f"检测错误: {err}")

    for ci in calib_imgs:
        err = ci.find_circle_indices(args.circle_interval, debug=False)
        if err:
            print(f"网格分配错误 ({ci.name}): {err}")

    valid_count = sum(1 for ci in calib_imgs if any(
        ok for _, _, ok, _ in ci.circle_array))
    if valid_count == 0:
        print("错误: 没有有效的网格分配结果")
        sys.exit(1)

    # 标定
    report, K, dist = calib.calibrate_camera(calib_imgs, "calib", debug=True)
    print(report)

    if K is not None and dist is not None:
        np.savez(args.output, camera_matrix=K, dist_coeffs=dist)
        print(f"\n已保存到 {args.output}")


if __name__ == "__main__":
    main()
