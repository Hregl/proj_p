"""阶段3: 标定板PnP → 求解相机外参"""
import sys, yaml, cv2, numpy as np, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def load_csv_points(filepath):
    pts, ids = [], []
    with open(filepath) as f:
        for line in f:
            if line.startswith('point_id'): continue
            parts = line.strip().split(',')
            if len(parts) >= 3:
                ids.append(parts[0])
                pts.append([float(parts[1]), float(parts[2])])
    return ids, np.array(pts, dtype=np.float64)

def main():
    import argparse
    p = argparse.ArgumentParser(description='标定板PnP位姿估计')
    p.add_argument('--config', default='configs/experiment_config.yaml')
    p.add_argument('--board-2d', default='annotations/board_2d/points.csv')
    p.add_argument('--board-3d', default='configs/board_points.yaml')
    p.add_argument('--output', '-o', default='output/board_pose.csv')
    args = p.parse_args()

    # 读配置
    with open(args.config) as f: exp = yaml.safe_load(f)
    K = np.array([[exp['calibration']['fx'],0,exp['calibration']['cx']],
                   [0,exp['calibration']['fy'],exp['calibration']['cy']],
                   [0,0,1]], dtype=np.float64)
    dist = np.array(exp['calibration']['dist'], dtype=np.float64)

    # 读3D点
    with open(args.board_3d) as f: board = yaml.safe_load(f)
    pts3d = np.array(list(board['points'].values()), dtype=np.float64)
    pt_ids_3d = list(board['points'].keys())

    # 读2D点
    pt_ids_2d, pts2d = load_csv_points(args.board_2d)

    # 建立对应
    obj, img = [], []
    for i, pid in enumerate(pt_ids_2d):
        if pid in board['points']:
            idx = pt_ids_3d.index(pid)
            obj.append(pts3d[idx])
            img.append(pts2d[i])

    if len(obj) < 4: print(f'对应点不足: {len(obj)}'); sys.exit(1)

    obj_arr = np.array(obj, dtype=np.float64)
    img_arr = np.array(img, dtype=np.float64)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_arr, img_arr, K, dist, flags=cv2.SOLVEPNP_EPNP,
        iterationsCount=200, reprojectionError=2.0, confidence=0.99)

    if not success: print('PnP失败'); sys.exit(1)

    R, _ = cv2.Rodrigues(rvec)
    n_inl = len(inliers) if inliers is not None else 0

    # 重投影误差
    proj, _ = cv2.projectPoints(obj_arr, rvec, tvec, K, dist)
    errs = np.linalg.norm(proj.reshape(-1,2) - img_arr, axis=1)
    rmse = float(np.sqrt(np.mean(errs**2)))

    print(f'标定板PnP: {n_inl}/{len(obj)} 内点, RMSE={rmse:.3f}px')
    print(f'R:\n{R}')
    print(f't (mm): {tvec.ravel()}')
    print(f'相机位置(board系): {-R.T @ tvec.ravel()}')

    # 保存
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        f.write('image_id,rvec_x,rvec_y,rvec_z,tvec_x,tvec_y,tvec_z,rmse_px,inlier_count\n')
        f.write(f'{Path(args.board_2d).stem},{rvec[0][0]:.6f},{rvec[1][0]:.6f},{rvec[2][0]:.6f},')
        f.write(f'{tvec[0][0]:.4f},{tvec[1][0]:.4f},{tvec[2][0]:.4f},{rmse:.4f},{n_inl}\n')
    print(f'-> {args.output}')

if __name__ == '__main__': main()
