"""在真实圆点网格图像上测试相机标定。"""
import sys
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sls_calib import CalibImage, Calibrator

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else str(DATA_DIR / "p1.png")
    circle_interval = float(sys.argv[2]) if len(sys.argv) > 2 else 35.0

    print(f"正在加载: {path}")
    img = cv2.imread(path)
    if img is None:
        print(f"错误: 无法读取 '{path}'")
        sys.exit(1)
    print(f"图像尺寸: {img.shape[1]} x {img.shape[0]}")

    calib_img = CalibImage(name="test", image=img.copy(), selected=True)
    calibrator = Calibrator()

    # 检测标记
    print("\n--- 正在检测标记 ---")
    err = calibrator.extract_circles([calib_img], only_selected=False,
                                     smooth=True, debug=True)
    if err:
        print(f"错误: {err}")
        sys.exit(1)
    print(f"检测到 {len(calib_img.circles)} 个圆")

    # 网格分配
    print(f"\n--- 网格分配 (间距={circle_interval}) ---")
    err = calib_img.find_circle_indices(circle_interval, debug=True,
                                         large_circle_threshold=0.78)
    if err:
        print(f"错误: {err}")
        sys.exit(1)

    # 显示网格
    print("\n--- 圆点阵列 (11x9 网格) ---")
    valid_count = 0
    for gy in range(9):
        line = ""
        for gx in range(11):
            _, _, ok, _ = calib_img.circle_array[gy * 11 + gx]
            if ok:
                valid_count += 1
                line += "O "
            else:
                line += ". "
        print(line)
    print(f"\n有效圆点数: {valid_count}/99")

    # 标定
    report, K, dist = calibrator.calibrate_camera([calib_img], "test",
                                                    debug=True)
    print(f"\n{report}")
    print("\n完成。")


if __name__ == "__main__":
    main()
