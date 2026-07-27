"""
P1: Pose repeatability statistics module.

Computes mean, std, min, max for yaw/pitch/roll across multiple
independent measurements of the same pose. Outputs in both degrees
and arcminutes.

Usage:
  python tools/evaluate_repeatability.py output/final_pose_*.csv \
      --group-name static_test_01 \
      --output output/repeatability_report.json
"""
import sys, csv, math, json
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_final_pose(filepath: str) -> dict:
    """Load a final pose CSV. Handles both old and new column formats."""
    with open(filepath) as f:
        reader = csv.DictReader(f)
        row = next(reader)

    # New format uses ground_rmse_px, old used board_rmse_px
    ground_rmse = float(
        row.get('ground_rmse_px', row.get('board_rmse_px', 0)))
    aircraft_rmse = float(row.get('aircraft_rmse_px', 0))
    ground_n_inl = int(float(row.get('ground_n_inliers', row.get('inlier_count', 0))))
    aircraft_n_inl = int(float(row.get('aircraft_n_inliers', 0)))

    return {
        'file': Path(filepath).stem,
        'yaw_deg': float(row['yaw_deg']),
        'pitch_deg': float(row['pitch_deg']),
        'roll_deg': float(row['roll_deg']),
        'yaw_arcmin': float(row['yaw_arcmin']),
        'pitch_arcmin': float(row['pitch_arcmin']),
        'roll_arcmin': float(row['roll_arcmin']),
        'pos_x': float(row['pos_x_mm']),
        'pos_y': float(row['pos_y_mm']),
        'pos_z': float(row['pos_z_mm']),
        'ground_rmse': ground_rmse,
        'aircraft_rmse': aircraft_rmse,
        'ground_n_inliers': ground_n_inl,
        'aircraft_n_inliers': aircraft_n_inl,
    }


def main():
    import argparse
    p = argparse.ArgumentParser(
        description='Compute pose repeatability statistics from multiple '
                    'final pose CSVs')
    p.add_argument('files', nargs='+',
                   help='Final pose CSV files (same pose, multiple measurements)')
    p.add_argument('--group-name', '-g', default='default',
                   help='Label for this pose group')
    p.add_argument('--output', '-o', default=None,
                   help='Output JSON report (also writes CSV if .csv extension)')
    p.add_argument('--max-ground-rmse', type=float, default=3.0,
                   help='Exclude measurements with ground RMSE above this '
                        'threshold')
    p.add_argument('--max-aircraft-rmse', type=float, default=5.0,
                   help='Exclude measurements with aircraft RMSE above this '
                        'threshold')
    args = p.parse_args()

    results = []
    excluded = []
    for fp in args.files:
        try:
            r = load_final_pose(fp)
        except Exception as e:
            print(f"  Skip {fp}: {e}")
            continue

        if r['ground_rmse'] > args.max_ground_rmse:
            excluded.append((r, 'ground_rmse'))
            continue
        if r['aircraft_rmse'] > args.max_aircraft_rmse:
            excluded.append((r, 'aircraft_rmse'))
            continue
        results.append(r)

    if not results:
        print("No valid measurements after filtering.")
        sys.exit(1)

    n_total = len(results) + len(excluded)

    yaws = np.array([r['yaw_deg'] for r in results])
    pitches = np.array([r['pitch_deg'] for r in results])
    rolls = np.array([r['roll_deg'] for r in results])
    g_rmse = np.array([r['ground_rmse'] for r in results])
    a_rmse = np.array([r['aircraft_rmse'] for r in results])

    # Yaw wrap-around check
    yaw_range = yaws.max() - yaws.min()
    yaw_wrapped = yaw_range > 300
    if yaw_wrapped:
        for i in range(len(yaws)):
            while yaws[i] - np.mean(yaws) > 180:
                yaws[i] -= 360
            while yaws[i] - np.mean(yaws) < -180:
                yaws[i] += 360

    def stats(arr):
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr, ddof=1)),
            'median': float(np.median(arr)),
            'mad': float(np.median(np.abs(arr - np.median(arr)))),
            'p95': float(np.percentile(np.abs(arr - np.mean(arr)), 95)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'range': float(np.max(arr) - np.min(arr)),
        }

    sy = stats(yaws)
    sp = stats(pitches)
    sr = stats(rolls)

    # Total rotation angle error (relative to mean rotation)
    from sls_calib.transforms import euler_to_R, rotation_angle_error
    R_mean = euler_to_R(sy['mean'], sp['mean'], sr['mean'])
    angle_errs = []
    for r in results:
        R_i = euler_to_R(r['yaw_deg'], r['pitch_deg'], r['roll_deg'])
        angle_errs.append(rotation_angle_error(R_mean, R_i))
    angle_errs = np.array(angle_errs)

    # Display
    n = len(results)
    fail_rate = (n_total - n) / n_total * 100 if n_total > 0 else 0
    print(f"\n{'='*60}")
    print(f"Repeatability Report: {args.group_name}")
    print(f"  Total samples: {n_total} | Valid: {n} | Excluded: {len(excluded)} "
          f"({fail_rate:.0f}% fail rate)")
    print(f"{'='*60}")

    if yaw_wrapped:
        print(f"  NOTE: yaw wrap detected (range={yaw_range:.0f} deg), "
              f"values unwrapped")

    print(f"\n{'':>12} {'mean':>8} {'std':>8} {'median':>8} {'MAD':>8} "
          f"{'p95':>8} {'min':>8} {'max':>8}")
    for label, s in [('yaw', sy), ('pitch', sp), ('roll', sr)]:
        print(f"  {label:>8}  {s['mean']:>7.3f}° {s['std']:>7.3f}° "
              f"{s['median']:>7.3f}° {s['mad']:>7.3f}° {s['p95']:>7.3f}° "
              f"{s['min']:>7.3f}° {s['max']:>7.3f}°")

    print(f"\n  {'yaw':>8}  {sy['std'] * 60:>5.1f} arcmin std (sample, ddof=1)")
    print(f"  {'pitch':>8}  {sp['std'] * 60:>5.1f} arcmin std")
    print(f"  {'roll':>8}  {sr['std'] * 60:>5.1f} arcmin std")
    total_angle_std = float(np.std(angle_errs, ddof=1)) * 60
    total_angle_p95 = float(np.percentile(angle_errs, 95)) * 60
    print(f"  {'total angle':>8}  {total_angle_std:>5.1f} arcmin std "
          f"(p95={total_angle_p95:.1f} arcmin)")

    # Per-measurement table
    print(f"\n{'Measurement':<30} {'yaw':>7} {'pitch':>7} {'roll':>7} "
          f"{'gRMSE':>7} {'aRMSE':>7}")
    print(f"{'-'*65}")
    for r in results:
        print(f"  {r['file']:<28} {r['yaw_deg']:>6.2f}° "
              f"{r['pitch_deg']:>6.2f}° {r['roll_deg']:>6.2f}° "
              f"{r['ground_rmse']:>6.2f} {r['aircraft_rmse']:>6.2f}")

    if excluded:
        print(f"\n  Excluded ({len(excluded)}):")
        for r, reason in excluded:
            print(f"    {r['file']}: {reason} "
                  f"(g={r['ground_rmse']:.1f}, a={r['aircraft_rmse']:.1f})")

    # Assessment
    std_arcmin = max(sy['std'], sp['std'], sr['std']) * 60
    if std_arcmin < 5:
        grade = "EXCELLENT (<5 arcmin)"
    elif std_arcmin < 10:
        grade = "GOOD (<10 arcmin)"
    elif std_arcmin < 30:
        grade = "FAIR (<30 arcmin)"
    else:
        grade = "POOR (>30 arcmin)"

    print(f"\n  Overall grade: {grade} "
          f"(worst-axis std = {std_arcmin:.1f} arcmin)")

    # --- Build report ---
    report = {
        'group': args.group_name,
        'stage': 'repeatability',
        'n_total': n_total,
        'n_valid': n,
        'n_excluded': len(excluded),
        'fail_rate_percent': round(fail_rate, 1),
        'yaw_wrapped': yaw_wrapped,
        'yaw_deg': sy,
        'pitch_deg': sp,
        'roll_deg': sr,
        'total_angle_error_deg': {
            'std': round(float(np.std(angle_errs, ddof=1)), 4),
            'p95': round(float(np.percentile(angle_errs, 95)), 4),
        },
        'total_angle_error_arcmin': {
            'std': round(total_angle_std, 1),
            'p95': round(total_angle_p95, 1),
        },
        'mean_ground_rmse_px': round(float(np.mean(g_rmse)), 2),
        'mean_aircraft_rmse_px': round(float(np.mean(a_rmse)), 2),
        'grade': grade.split(' ')[0],
        'excluded': [{'file': r['file'], 'reason': reason}
                    for r, reason in excluded],
        'per_measurement': results,
    }

    # --- Export ---
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.suffix == '.json':
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"  -> {out_path}")
        elif out_path.suffix == '.csv':
            write_header = not out_path.exists()
            with open(out_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        'group', 'n', 'yaw_mean_deg', 'yaw_std_deg',
                        'yaw_std_arcmin', 'pitch_mean_deg', 'pitch_std_deg',
                        'pitch_std_arcmin', 'roll_mean_deg', 'roll_std_deg',
                        'roll_std_arcmin', 'mean_ground_rmse',
                        'mean_aircraft_rmse', 'grade'])
                writer.writerow([
                    args.group_name, n,
                    round(sy['mean'], 4), round(sy['std'], 4),
                    round(sy['std'] * 60, 1),
                    round(sp['mean'], 4), round(sp['std'], 4),
                    round(sp['std'] * 60, 1),
                    round(sr['mean'], 4), round(sr['std'], 4),
                    round(sr['std'] * 60, 1),
                    round(float(np.mean(g_rmse)), 2),
                    round(float(np.mean(a_rmse)), 2),
                    grade.split(' ')[0]])
            print(f"  -> {out_path}")
        else:
            # Default: save as JSON
            json_path = out_path.with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"  -> {json_path}")


if __name__ == '__main__':
    main()
