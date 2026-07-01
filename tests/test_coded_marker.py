"""Test coded marker generation and detection."""
import sys
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sls_calib import CodedMarkerDetector, generate_marker_sheet


def main():
    # Generate a synthetic marker sheet and detect on it
    print("Generating synthetic test image...")
    sheet = generate_marker_sheet("4x4_50", pixel_size=200, ids=list(range(8)))
    cv2.imwrite("aruco_sheet.png", sheet)
    print("Wrote aruco_sheet.png (8 markers, 4x2 grid)")

    img = cv2.imread("aruco_sheet.png")
    detector = CodedMarkerDetector(dict_name="4x4_50", refine_corners=True)
    markers, err = detector.detect(img, debug=True)

    if err:
        print(f"Error: {err}")
        sys.exit(1)

    print(f"\nFound {len(markers)} ArUco marker(s):")
    for m_id, corners, center, side in markers:
        print(f"  ID={m_id:3d}  center=({center[0]:7.1f}, {center[1]:7.1f})  "
              f"side={side:.1f}px")

    # Visualize and save
    annotated = CodedMarkerDetector.draw(img, markers)
    cv2.imwrite("aruco_detected.png", annotated)
    print("\nWrote aruco_detected.png (detection visualization)")

    # Verify all 8 markers detected
    detected_ids = {m[0] for m in markers}
    expected = set(range(8))
    if detected_ids == expected:
        print("\nPASS: All 8 markers detected with correct IDs")
    else:
        print(f"\nFAIL: Expected {expected}, got {detected_ids}")
        sys.exit(1)


if __name__ == "__main__":
    main()
