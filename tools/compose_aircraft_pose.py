"""阶段5: 合成飞机相对标定板的最终姿态 (G_R_B = inv(G_R_C) * C_R_B)"""
import sys, yaml, cv2, numpy as np, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def rotation_to_euler(R):
    """ZYX欧拉角: yaw(Z), pitch(Y), roll(X)"""
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        rx = math.atan2(R[2,1], R[2,2])
        ry = math.atan2(-R[2,0], sy)
        rz = math.atan2(R[1,0], R[0,0])
    else:
        rx = math.atan2(-R[1,2], R[1,1])
        ry = math.atan2(-R[2,0], sy)
        rz = 0.0
    return np.degrees(rx), np.degrees(ry), np.degrees(rz)

def load_csv_pose(filepath):
    with open(filepath) as f:
        for line in f:
            if line.startswith('image_id'): continue
            parts = line.strip().split(',')
            if len(parts) >= 7:
                rvec = np.array([float(parts[1]),float(parts[2]),float(parts[3])])
                tvec = np.array([float(parts[4]),float(parts[5]),float(parts[6])])
                rmse = float(parts[7]) if len(parts) > 7 else 0
                R, _ = cv2.Rodrigues(rvec)
                return R, tvec, rmse
    return None, None, 0

def main():
    import argparse
    p = argparse.ArgumentParser(description='合成飞机相对标定板参考系的最终姿态')
    p.add_argument('--config', default='configs/experiment_config.yaml')
    p.add_argument('--board-pose', required=True, help='标定板PnP结果CSV')
    p.add_argument('--aircraft-pose', required=True, help='飞机PnP结果CSV')
    p.add_argument('--output', '-o', default='output/final_pose.csv')
    args = p.parse_args()

    with open(args.config) as f: exp = yaml.safe_load(f)

    # board_to_camera: C_T_G
    C_R_G, C_t_G, board_rmse = load_csv_pose(args.board_pose)
    if C_R_G is None: print('标定板PnP结果无效'); sys.exit(1)

    # aircraft_to_camera: C_T_B
    C_R_B, C_t_B, ac_rmse = load_csv_pose(args.aircraft_pose)
    if C_R_B is None: print('飞机PnP结果无效'); sys.exit(1)

    # 相机在标定板系中的位姿: G_R_C = inv(C_R_G)
    G_R_C = np.linalg.inv(C_R_G)  # 使用真逆(处理PnP结果的非严格正交性)
    G_t_C = -G_R_C @ C_t_G        # 相机原点在G系中的位置
    # 飞机在标定板系中的姿态: G_R_B = G_R_C * C_R_B
    G_R_B = G_R_C @ C_R_B
    G_t_B = G_t_C + G_R_C @ C_t_B  # 飞机原点在G系中的位置

    yaw, pitch, roll = rotation_to_euler(G_R_B)

    # 万向节死锁检测
    sy = math.sqrt(G_R_B[0,0]**2 + G_R_B[1,0]**2)
    gimbal_warning = ''
    if sy < 1e-3:
        gimbal_warning = ' [警告: 万向节死锁, yaw/roll不可区分]'

    print(f'=== 最终姿态 (飞机 in 标定板参考系) ===')
    print(f'yaw:   {yaw:.4f} deg  ({yaw*60:.2f} arcmin){gimbal_warning}')
    print(f'pitch: {pitch:.4f} deg  ({pitch*60:.2f} arcmin)')
    print(f'roll:  {roll:.4f} deg  ({roll*60:.2f} arcmin)')
    print(f'位置(G系): ({G_t_B[0]:.1f}, {G_t_B[1]:.1f}, {G_t_B[2]:.1f}) mm')
    print(f'board RMSE: {board_rmse:.4f} px')
    print(f'aircraft RMSE: {ac_rmse:.4f} px')

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        f.write('yaw_deg,pitch_deg,roll_deg,yaw_arcmin,pitch_arcmin,roll_arcmin,'
                'pos_x_mm,pos_y_mm,pos_z_mm,board_rmse_px,aircraft_rmse_px,gimbal_warning\n')
        f.write(f'{yaw:.6f},{pitch:.6f},{roll:.6f},{yaw*60:.4f},{pitch*60:.4f},{roll*60:.4f},')
        f.write(f'{G_t_B[0]:.2f},{G_t_B[1]:.2f},{G_t_B[2]:.2f},{board_rmse:.4f},{ac_rmse:.4f},'
                f'{1 if gimbal_warning else 0}\n')
    print(f'-> {args.output}')

if __name__ == '__main__': main()
