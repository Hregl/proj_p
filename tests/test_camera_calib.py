"""Test camera calibration on a real dot-grid image."""
import sys
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sls_calib import CalibImage, Calibrator

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else str(DATA_DIR / "p1.png")
    circle_interval = float(sys.argv[2]) if len(sys.argv) > 2 else 35.0

    print(f"Loading: {path}")
    img = cv2.imread(path)
    if img is None:
        print(f"Error: cannot read '{path}'")
        sys.exit(1)
    print(f"Image size: {img.shape[1]} x {img.shape[0]}")

    calib_img = CalibImage(name="test", image=img.copy(), selected=True)
    calibrator = Calibrator()

    # Detect markers
    print("\n--- Detecting markers ---")
    err = calibrator.extract_circles([calib_img], only_selected=False,
                                     smooth=True, debug=True)
    if err:
        print(f"Error: {err}")
        sys.exit(1)
    print(f"Detected {len(calib_img.circles)} circles")

    # Grid assignment
    print(f"\n--- Grid assignment (interval={circle_interval}) ---")
    err = calib_img.find_circle_indices(circle_interval, debug=True,
                                         large_circle_threshold=0.78)
    if err:
        print(f"Error: {err}")
        sys.exit(1)

    # Show grid
    print("\n--- Circle array (11x9 grid) ---")
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
    print(f"\nValid circles: {valid_count}/99")

    # Calibrate
    report, K, dist = calibrator.calibrate_camera([calib_img], "test",
                                                    debug=True)
    print(f"\n{report}")
    print("\nDone.")


if __name__ == "__main__":
    main()
