"""在标定图像上测试 SLSMarkerDetector。"""
import sys
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sls_calib import SLSMarkerDetector

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else str(DATA_DIR / "p1.png")
    print(f"正在测试: {path}")

    img = cv2.imread(path)
    if img is None:
        print(f"错误: 无法读取图像 '{path}'")
        sys.exit(1)
    print(f"图像尺寸: {img.shape[1]} x {img.shape[0]}")

    detector = SLSMarkerDetector()
    markers, err = detector.detectMarkers(img, smooth=False, debug=True)
    if err:
        print(f"错误: {err}")
    else:
        print(f"\n找到 {len(markers)} 个标记:")
        for i, ((cx, cy), area) in enumerate(markers):
            print(f"  [{i}] 中心=({cx:.3f}, {cy:.3f})  面积={area:.3f}")


if __name__ == "__main__":
    main()
