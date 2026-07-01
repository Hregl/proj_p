"""Test SLSMarkerDetector on a calibration image."""
import sys
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sls_calib import SLSMarkerDetector

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else str(DATA_DIR / "p1.png")
    print(f"Testing on: {path}")

    img = cv2.imread(path)
    if img is None:
        print(f"Error: could not read image '{path}'")
        sys.exit(1)
    print(f"Image size: {img.shape[1]} x {img.shape[0]}")

    detector = SLSMarkerDetector()
    markers, err = detector.detectMarkers(img, smooth=False, debug=True)
    if err:
        print(f"Error: {err}")
    else:
        print(f"\nFound {len(markers)} marker(s):")
        for i, ((cx, cy), area) in enumerate(markers):
            print(f"  [{i}] center=({cx:.3f}, {cy:.3f})  area={area:.3f}")


if __name__ == "__main__":
    main()
