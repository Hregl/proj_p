"""
P1: Pose repeatability statistics module.

Computes mean, std, min, max for yaw/pitch/roll across multiple
independent measurements of the same pose. Outputs in both degrees
and arcminutes.

Usage:
  python tools/evaluate_repeatability.py output/*_final.csv
"""
import sys, csv, math
import numpy as np
from pathlib import Path

def load_final_pose(filepath: str) -> dict:
    """Load a final pose CSV. Returns dict with yaw/pitch/roll/rmse/etc."""
    with open(filepath) as f:
        reader = csv.DictReader(f)
        row = next(reader)
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
        'board_rmse': float(row['board_rmse_px']),
        'aircraft_rmse': float(row['aircraft_rmse_px']),
    }

def main():
    import argparse
    p = argparse.ArgumentParser(
        description='Compute pose repeatability statistics from multiple final pose CSVs')
    p.add_argument('files', nargs='+', help='Final pose CSV files (same pose, multiple measurements)')
    p.add_argument('--group-name', '-g', default='default',
                   help='Label for this pose group')
    p.add_argument('--output', '-o', default=None,
                   help='Output CSV (appends if exists)')
    p.add_argument('--max-board-rmse', type=float, default=3.0,
                   help='Exclude measurements with board RMSE above this threshold')
    p.add_argument('--max-aircraft-rmse', type=float, default=5.0,
                   help='Exclude measurements with aircraft RMSE above this threshold')
    args = p.parse_args()

    results = []
    excluded = []
    for fp in args.files:
        try:
            r = load_final_pose(fp)
        except Exception as e:
            print(f"  Skip {fp}: {e}")
            continue

        if r['board_rmse'] > args.max_board_rmse:
            excluded.append((r, 'board_rmse'))
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
    b_rmse = np.array([r['board_rmse'] for r in results])
    a_rmse = np.array([r['aircraft_rmse'] for r in results])

    # Yaw wrap-around check: detect if yaw values span > 300 deg
    yaw_range = yaws.max() - yaws.min()
    yaw_wrapped = yaw_range > 300
    if yaw_wrapped:
        # Unwrap: shift values > 0 to negative if mean is near 0
        for i in range(len(yaws)):
            while yaws[i] - np.mean(yaws) > 180:
                yaws[i] -= 360
            while yaws[i] - np.mean(yaws) < -180:
                yaws[i] += 360

    def stats(arr, name):
        return {
            'mean': np.mean(arr),
            'std': np.std(arr, ddof=1),  # sample std
            'median': np.median(arr),
            'mad': np.median(np.abs(arr - np.median(arr))),
            'p95': np.percentile(np.abs(arr - np.mean(arr)), 95),
            'min': np.min(arr), 'max': np.max(arr),
            'range': np.max(arr) - np.min(arr),
        }

    sy = stats(yaws, 'yaw')
    sp = stats(pitches, 'pitch')
    sr = stats(rolls, 'roll')

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
        print(f"  NOTE: yaw wrap detected (range={yaw_range:.0f} deg), values unwrapped")

    print(f"\n{'':>12} {'mean':>8} {'std':>8} {'median':>8} {'MAD':>8} {'p95':>8} {'min':>8} {'max':>8}")
    for label, s in [('yaw', sy), ('pitch', sp), ('roll', sr)]:
        print(f"  {label:>8}  {s['mean']:>7.3f}° {s['std']:>7.3f}° "
              f"{s['median']:>7.3f}° {s['mad']:>7.3f}° {s['p95']:>7.3f}° "
              f"{s['min']:>7.3f}° {s['max']:>7.3f}°")

    print(f"\n  {'yaw':>8}  {sy['std']*60:>5.1f} arcmin std (sample, ddof=1)")
    print(f"  {'pitch':>8}  {sp['std']*60:>5.1f} arcmin std")
    print(f"  {'roll':>8}  {sr['std']*60:>5.1f} arcmin std")
    print(f"  {'total angle':>8}  {np.std(angle_errs, ddof=1)*60:>5.1f} arcmin std "
          f"(p95={np.percentile(angle_errs,95)*60:.1f} arcmin)")

    # Per-measurement table
    print(f"\n{'Measurement':<30} {'yaw':>7} {'pitch':>7} {'roll':>7} {'bRMSE':>7} {'aRMSE':>7}")
    print(f"{'-'*65}")
    for r in results:
        print(f"  {r['file']:<28} {r['yaw_deg']:>6.2f}° {r['pitch_deg']:>6.2f}° "
              f"{r['roll_deg']:>6.2f}° {r['board_rmse']:>6.2f} {r['aircraft_rmse']:>6.2f}")

    if excluded:
        print(f"\n  Excluded ({len(excluded)}):")
        for r, reason in excluded:
            print(f"    {r['file']}: {reason} (b={r['board_rmse']:.1f}, a={r['aircraft_rmse']:.1f})")

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

    print(f"\n  Overall grade: {grade} (worst-axis std = {std_arcmin:.1f} arcmin)")

    # Export
    if args.output:
        write_header = not Path(args.output).exists()
        with open(args.output, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['group','n','yaw_mean_deg','yaw_std_deg','yaw_std_arcmin',
                                'pitch_mean_deg','pitch_std_deg','pitch_std_arcmin',
                                'roll_mean_deg','roll_std_deg','roll_std_arcmin',
                                'mean_board_rmse','mean_aircraft_rmse','grade'])
            writer.writerow([args.group_name, n,
                            round(sy['mean'], 4), round(sy['std'], 4), round(sy['std']*60, 1),
                            round(sp['mean'], 4), round(sp['std'], 4), round(sp['std']*60, 1),
                            round(sr['mean'], 4), round(sr['std'], 4), round(sr['std']*60, 1),
                            round(float(np.mean(b_rmse)), 2),
                            round(float(np.mean(a_rmse)), 2),
                            grade.split(' ')[0]])
        print(f"  -> {args.output}")

if __name__ == '__main__':
    main()
