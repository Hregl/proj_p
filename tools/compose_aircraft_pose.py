"""Stage 4: Compose final aircraft pose in ground frame (G_T_B).

Composes ground pose (C_T_G) and aircraft pose (C_T_B) to obtain
the aircraft pose in the ground coordinate system G.

Uses sls_calib.transforms as the single source of truth for:
  - compose_G_T_B(): pose composition
  - R_to_euler(): ZYX Euler (yaw, pitch, roll) — correct axis order
"""

import sys, yaml, cv2, numpy as np, math, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sls_calib.transforms import compose_G_T_B, R_to_euler, rotation_angle_error


def load_csv_pose(filepath):
    """Load rvec/tvec from a PnP CSV (board/ground or aircraft pose).

    Handles both old and new CSV formats with varying column counts.
    Returns (R, t, rmse, n_inliers).
    """
    with open(filepath) as f:
        for line in f:
            if line.startswith('image_id'):
                continue
            parts = line.strip().split(',')
            if len(parts) >= 7:
                try:
                    rvec = np.array([float(parts[1]), float(parts[2]),
                                     float(parts[3])])
                    tvec = np.array([float(parts[4]), float(parts[5]),
                                     float(parts[6])])
                    rmse = 0.0
                    n_inl = 0
                    for i in range(7, len(parts)):
                        try:
                            val = float(parts[i])
                            if rmse == 0.0:
                                rmse = val
                            elif n_inl == 0:
                                n_inl = int(val)
                        except ValueError:
                            continue
                    R, _ = cv2.Rodrigues(rvec)
                    return R, tvec, rmse, n_inl
                except (ValueError, cv2.error):
                    continue
    return None, None, 0.0, 0


def main():
    import argparse
    p = argparse.ArgumentParser(
        description='Stage 4: Compose final aircraft pose in ground frame (G_T_B)')
    p.add_argument('--config', default='configs/experiment_config.yaml')
    p.add_argument('--ground-pose', required=True,
                   help='Ground PnP result CSV (C_T_G)')
    p.add_argument('--aircraft-pose', required=True,
                   help='Aircraft PnP result CSV (C_T_B)')
    p.add_argument('--output', '-o', default='output/final_pose.csv')
    p.add_argument('--output-json', default=None,
                   help='Optional JSON output with full pose + quality fields')
    args = p.parse_args()

    # --- Load poses ---
    C_R_G, C_t_G, ground_rmse, ground_n_inl = load_csv_pose(args.ground_pose)
    if C_R_G is None:
        print(f'Ground PnP result invalid: {args.ground_pose}')
        sys.exit(1)

    C_R_B, C_t_B, ac_rmse, ac_n_inl = load_csv_pose(args.aircraft_pose)
    if C_R_B is None:
        print(f'Aircraft PnP result invalid: {args.aircraft_pose}')
        sys.exit(1)

    # --- Compose: G_T_B = inv(C_T_G) * C_T_B ---
    # Uses the canonical implementation from sls_calib.transforms
    G_R_B, G_t_B = compose_G_T_B(C_R_G, C_t_G, C_R_B, C_t_B)

    # --- Extract Euler angles (correct order: yaw, pitch, roll) ---
    yaw, pitch, roll = R_to_euler(G_R_B)

    # Gimbal lock detection
    sy = math.sqrt(G_R_B[0, 0]**2 + G_R_B[1, 0]**2)
    gimbal_warning = ''
    if sy < 1e-3:
        gimbal_warning = ' [Warning: gimbal lock, yaw/roll indistinguishable]'

    # --- Output ---
    print(f'\n=== Final pose (aircraft in ground G frame) ===')
    print(f'yaw:   {yaw:.4f} deg  ({yaw * 60:.2f} arcmin){gimbal_warning}')
    print(f'pitch: {pitch:.4f} deg  ({pitch * 60:.2f} arcmin)')
    print(f'roll:  {roll:.4f} deg  ({roll * 60:.2f} arcmin)')
    print(f'Position (G frame): ({G_t_B[0]:.1f}, {G_t_B[1]:.1f}, '
          f'{G_t_B[2]:.1f}) mm')
    print(f'ground RMSE: {ground_rmse:.4f} px (n_inl={ground_n_inl})')
    print(f'aircraft RMSE: {ac_rmse:.4f} px (n_inl={ac_n_inl})')

    # --- Save CSV ---
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write('yaw_deg,pitch_deg,roll_deg,'
                'yaw_arcmin,pitch_arcmin,roll_arcmin,'
                'pos_x_mm,pos_y_mm,pos_z_mm,'
                'ground_rmse_px,aircraft_rmse_px,'
                'ground_n_inliers,aircraft_n_inliers,'
                'gimbal_warning\n')
        f.write(f'{yaw:.6f},{pitch:.6f},{roll:.6f},'
                f'{yaw * 60:.4f},{pitch * 60:.4f},{roll * 60:.4f},'
                f'{G_t_B[0]:.2f},{G_t_B[1]:.2f},{G_t_B[2]:.2f},'
                f'{ground_rmse:.4f},{ac_rmse:.4f},'
                f'{ground_n_inl},{ac_n_inl},'
                f'{1 if gimbal_warning else 0}\n')
    print(f'-> {args.output}')

    # --- Save JSON (optional) ---
    if args.output_json:
        json_data = {
            'coordinate_system': 'G',
            'unit': 'mm',
            'euler_convention': 'ZYX (yaw-pitch-roll)',
            'yaw_deg': round(yaw, 6),
            'pitch_deg': round(pitch, 6),
            'roll_deg': round(roll, 6),
            'yaw_arcmin': round(yaw * 60, 4),
            'pitch_arcmin': round(pitch * 60, 4),
            'roll_arcmin': round(roll * 60, 4),
            'position_G_mm': [round(float(v), 2) for v in G_t_B],
            'G_R_B': [[round(float(v), 6) for v in row] for row in G_R_B],
            'ground_rmse_px': round(ground_rmse, 4),
            'aircraft_rmse_px': round(ac_rmse, 4),
            'ground_n_inliers': ground_n_inl,
            'aircraft_n_inliers': ac_n_inl,
            'gimbal_lock': bool(gimbal_warning),
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f'-> {args.output_json}')


if __name__ == '__main__':
    main()
