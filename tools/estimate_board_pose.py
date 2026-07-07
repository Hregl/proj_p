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
    try:
        with open(args.config) as f: exp = yaml.safe_load(f)
    except FileNotFoundError:
        print(f'找不到配置文件: {args.config}'); sys.exit(1)

    cal = exp.get('calibration', {})
    try:
        K = np.array([[cal['fx'],0,cal['cx']],
                       [0,cal['fy'],cal['cy']],
                       [0,0,1]], dtype=np.float64)
        dist = np.array(cal['dist'], dtype=np.float64)
    except KeyError as e:
        print(f'配置文件缺少必要字段: {e} (需要fx,fy,cx,cy,dist)'); sys.exit(1)
    # 验证K矩阵合理性
    if K[0,0] <= 0 or K[1,1] <= 0:
        print(f'K矩阵异常: fx={K[0,0]}, fy={K[1,1]}'); sys.exit(1)

    # 读3D点
    try:
        with open(args.board_3d) as f: board = yaml.safe_load(f)
    except FileNotFoundError:
        print(f'找不到标定板3D点文件: {args.board_3d}'); sys.exit(1)
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

    if not success:
        print(f'PnP失败: {len(obj_arr)}个点,RANSAC未收敛'); sys.exit(1)

    R, _ = cv2.Rodrigues(rvec)
    n_inl = len(inliers) if inliers is not None else 0

    # 重投影误差(仅在内点上计算)
    if inliers is not None and len(inliers) > 0:
        inl_mask = inliers.ravel().astype(bool)
        obj_inl = obj_arr[inl_mask]; img_inl = img_arr[inl_mask]
        proj, _ = cv2.projectPoints(obj_inl, rvec, tvec, K, dist)
    else:
        obj_inl, img_inl = obj_arr, img_arr
        proj, _ = cv2.projectPoints(obj_inl, rvec, tvec, K, dist)
    errs = np.linalg.norm(proj.reshape(-1,2) - img_inl, axis=1)
    rmse = float(np.sqrt(np.mean(errs**2)))

    print(f'标定板PnP: {n_inl}/{len(obj)} 内点, RMSE={rmse:.3f}px')
    print(f'R:\n{R}')
    print(f't (mm): {tvec.ravel()}')
    print(f'相机位置(board系): {-R.T @ tvec.ravel()}')

    # 保存
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    rv = rvec.ravel()
    tv = tvec.ravel()
    with open(args.output, 'w') as f:
        f.write('image_id,rvec_x,rvec_y,rvec_z,tvec_x,tvec_y,tvec_z,rmse_px,inlier_count\n')
        f.write(f'{Path(args.board_2d).stem},{rv[0]:.6f},{rv[1]:.6f},{rv[2]:.6f},')
        f.write(f'{tv[0]:.4f},{tv[1]:.4f},{tv[2]:.4f},{rmse:.4f},{n_inl}\n')
    print(f'-> {args.output}')

if __name__ == '__main__': main()
