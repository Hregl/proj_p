"""测试编码标记的生成与检测。"""
import sys
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sls_calib import CodedMarkerDetector, generate_marker_sheet


def main():
    # 生成合成标记页并在其上检测
    from pathlib import Path
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    print("正在生成合成测试图像...")
    sheet = generate_marker_sheet("4x4_50", pixel_size=200, ids=list(range(8)))
    cv2.imwrite(str(out_dir / "aruco_sheet.png"), sheet)
    print(f"已写入 {out_dir / 'aruco_sheet.png'} (8 个标记, 4x2 网格)")

    img = cv2.imread(str(out_dir / "aruco_sheet.png"))
    detector = CodedMarkerDetector(dict_name="4x4_50", refine_corners=True)
    markers, err = detector.detect(img, debug=True)

    if err:
        print(f"错误: {err}")
        sys.exit(1)

    print(f"\n找到 {len(markers)} 个 ArUco 标记:")
    for m_id, corners, center, side in markers:
        print(f"  ID={m_id:3d}  中心=({center[0]:7.1f}, {center[1]:7.1f})  "
              f"边长={side:.1f}px")

    # 可视化并保存
    annotated = CodedMarkerDetector.draw(img, markers)
    cv2.imwrite(str(out_dir / "aruco_detected.png"), annotated)
    print(f"\n已写入 {out_dir / 'aruco_detected.png'} (检测可视化)")

    # 验证所有 8 个标记均已检测到
    detected_ids = {m[0] for m in markers}
    expected = set(range(8))
    if detected_ids == expected:
        print("\n通过: 全部 8 个标记均已检测到，ID 正确")
    else:
        print(f"\n失败: 期望 {expected}, 实际得到 {detected_ids}")
        sys.exit(1)


if __name__ == "__main__":
    main()
