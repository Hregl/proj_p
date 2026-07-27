"""Stage 1: Camera intrinsic calibration.

Calibrates a camera using an SLS circular-dot target. Outputs:
  - camera_intrinsics.npz       (K, dist, image size)
  - calibration_report.json     (per-image metrics, overall stats)
  - calibration_report.csv      (per-image table)
  - reprojection_residuals.png  (visualization, if requested)

Usage:
    python tools/run_calibration.py data/calib_001.png --circle-interval 25 -o output/session_01/
"""

import sys, json, csv, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2
import numpy as np
from sls_calib import CalibImage, Calibrator


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Stage 1: Camera intrinsic calibration (SLS circular-dot target)")
    parser.add_argument("images", nargs="+", help="Calibration target images")
    parser.add_argument("--circle-interval", type=float, default=25.0,
                        help="Physical spacing between circles (mm) (default: 25)")
    parser.add_argument("--output", "-o", default="output/calibration",
                        help="Output directory for intrinsics + reports")
    parser.add_argument("--smooth", action="store_true",
                        help="Apply Gaussian blur before detection")
    parser.add_argument("--draw-residuals", action="store_true",
                        help="Generate reprojection residual visualization")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load images ---
    calib_imgs = []
    per_image_info = []  # Track per-image metadata for report
    for i, p in enumerate(args.images):
        img = cv2.imread(p)
        if img is None:
            print(f"Warning: cannot read '{p}', skipping")
            per_image_info.append({
                'index': i, 'file': Path(p).name, 'status': 'skipped',
                'reason': 'cannot read image'
            })
            continue
        calib_imgs.append(CalibImage(name=f"calib_{i}", image=img, selected=True))
        per_image_info.append({
            'index': i, 'file': Path(p).name, 'status': 'loaded',
            'width': img.shape[1], 'height': img.shape[0]
        })

    if len(calib_imgs) < 1:
        print("Error: no valid images loaded")
        sys.exit(1)

    print(f"Loaded {len(calib_imgs)} images")

    # --- Detect circles ---
    calib = Calibrator()
    err = calib.extract_circles(calib_imgs, only_selected=True,
                                 smooth=args.smooth, debug=True)
    if err:
        print(f"Detection error: {err}")

    # --- Assign grid indices ---
    for ci in calib_imgs:
        ci_err = ci.find_circle_indices(args.circle_interval, debug=False)
        if ci_err:
            print(f"Grid assignment error ({ci.name}): {ci_err}")

    valid_count = sum(1 for ci in calib_imgs if any(
        ok for _, _, ok, _ in ci.circle_array))
    if valid_count == 0:
        print("Error: no valid grid assignment results")
        sys.exit(1)

    # --- Calibrate ---
    report, K, dist = calib.calibrate_camera(calib_imgs, "calib", debug=True)
    print(report)

    if K is None or dist is None:
        print("Error: calibration failed")
        sys.exit(1)

    # --- Save intrinsics ---
    npz_path = out_dir / "stage_01_intrinsics.npz"
    h, w = calib_imgs[0].image.shape[:2]
    np.savez(npz_path,
             camera_matrix=K, dist_coeffs=dist,
             image_width=w, image_height=h)
    print(f"\nSaved: {npz_path}")

    # --- Per-image reprojection error ---
    # Re-project using calibrated intrinsics
    per_image_metrics = []
    total_points = 0
    total_err_sq = 0.0
    rms_px = 0.0  # from calibrate_camera return value

    # Parse the RMS from the report string
    for line in report.split('\n'):
        if '重投影误差' in line or 'reprojection' in line.lower():
            try:
                rms_px = float(line.split(':')[-1].strip())
            except ValueError:
                pass

    for i, ci in enumerate(calib_imgs):
        n_detected = len(ci.circles)
        n_assigned = sum(1 for _, _, ok, _ in ci.circle_array if ok)
        grid_err = ci.find_circle_indices(args.circle_interval, debug=False)

        img_metrics = {
            'index': i,
            'name': ci.name,
            'detected_circles': n_detected,
            'assigned_circles': n_assigned,
            'used': n_assigned >= 10,
            'grid_assignment_error': grid_err if grid_err else '',
        }
        per_image_metrics.append(img_metrics)
        if n_assigned >= 10:
            total_points += n_assigned

    # --- Build calibration report ---
    report_data = {
        'session_id': out_dir.name,
        'stage': 'camera_intrinsics',
        'status': 'pass' if rms_px < 2.0 else 'warning',
        'inputs': {
            'n_images_loaded': len(calib_imgs),
            'n_images_total': len(args.images),
            'circle_interval_mm': args.circle_interval,
            'smooth': args.smooth,
        },
        'outputs': {
            'intrinsics_file': str(npz_path),
            'image_width': w,
            'image_height': h,
        },
        'metrics': {
            'camera_matrix': [[round(float(v), 4) for v in row] for row in K],
            'dist_coeffs': [round(float(v), 6) for v in dist.ravel()],
            'overall_rms_px': round(rms_px, 4),
            'effective_images': valid_count,
        },
        'thresholds': {
            'max_rms_px': 2.0,
        },
        'per_image_metrics': per_image_metrics,
        'warnings': [],
        'failed_items': [],
    }

    # Check for poor images
    for m in per_image_metrics:
        if not m['used'] and m['detected_circles'] > 0:
            report_data['warnings'].append(
                f"{m['name']}: {m['detected_circles']} circles detected "
                f"but only {m['assigned_circles']} assigned")
            report_data['failed_items'].append(m['name'])

    # --- Save JSON report ---
    report_json = out_dir / "stage_01_calibration_report.json"
    with open(report_json, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"Report: {report_json}")

    # --- Save CSV report ---
    report_csv = out_dir / "stage_01_calibration_report.csv"
    with open(report_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'name', 'detected_circles', 'assigned_circles',
                         'used', 'grid_error'])
        for m in per_image_metrics:
            writer.writerow([m['index'], m['name'], m['detected_circles'],
                            m['assigned_circles'], m['used'],
                            m['grid_assignment_error']])
    print(f"CSV:    {report_csv}")

    # --- Residual visualization ---
    if args.draw_residuals:
        try:
            from sls_calib.transforms import project_points
            # Collect all 3D points and project them
            all_obj = []
            all_img = []
            for ci in calib_imgs:
                for (px, py), (wx, wy, wz), ok, _ in ci.circle_array:
                    if ok:
                        all_obj.append([wx, wy, wz])
                        all_img.append([px, py])

            if all_obj:
                all_obj = np.array(all_obj, dtype=np.float64)
                all_img = np.array(all_img, dtype=np.float64)

                # Use zero distortion for residual arrow visualization
                rvec_zero = np.zeros(3)
                tvec_zero = np.zeros(3)
                proj, _ = cv2.projectPoints(all_obj, rvec_zero, tvec_zero, K, None)
                proj = proj.reshape(-1, 2)
                errs = np.linalg.norm(proj - all_img, axis=1)

                # Create scatter plot using OpenCV
                vis_h, vis_w = 800, 1200
                vis = np.ones((vis_h, vis_w, 3), dtype=np.uint8) * 255

                # Plot residual histogram
                bin_w = 40
                bins = np.arange(0, max(errs.max() + 0.5, 2.0), 0.1)
                hist, _ = np.histogram(errs, bins=bins)
                max_hist = hist.max() if hist.max() > 0 else 1

                for j, (count, edge) in enumerate(zip(hist, bins[:-1])):
                    bar_h = int(count / max_hist * (vis_h - 100))
                    cv2.rectangle(vis,
                                  (50 + j * bin_w // 10, vis_h - 50 - bar_h),
                                  (50 + (j + 1) * bin_w // 10, vis_h - 50),
                                  (100, 100, 255), -1)

                cv2.putText(vis, f"Reprojection Residuals (RMS={rms_px:.3f}px)",
                           (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(vis, "Error (px) →", (50, vis_h - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                res_path = out_dir / "stage_01_reprojection_residuals.png"
                cv2.imwrite(str(res_path), vis)
                print(f"Residuals: {res_path}")
        except Exception as e:
            print(f"Residual viz failed: {e}")

    print(f"\n{'='*60}")
    print(f"Stage 1 Summary:")
    print(f"  Camera: {w}x{h}, K = [{K[0,0]:.1f}, {K[1,1]:.1f}]")
    print(f"  RMS: {rms_px:.4f} px")
    print(f"  Valid images: {valid_count}/{len(calib_imgs)}")
    print(f"  Output: {npz_path}")


if __name__ == "__main__":
    main()
