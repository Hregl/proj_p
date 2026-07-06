"""
多视图三角测量飞机标志点三维坐标。

利用标定板作为已知参考（每张图PnP得到相机位姿），
在第一张图上标注飞机标志点后，半自动跨视图跟踪+三角测量，
直接输出飞机标志点在标定板参考系(G)下的三维坐标。

使用方式:
  python tools/triangulate_aircraft_points.py data/scene/*.png

流程:
  1. 每张图检测标定板 → PnP → 相机位姿(在G系中)
  2. 第一张图打开GUI → 用户点击标注飞机标志点
  3. 自动用光流法跟踪到后续视图
  4. 多视图三角测量 → 3D坐标
  5. 统计重投影误差 → 输出 aircraft_points.yaml
"""

import sys, yaml, cv2, numpy as np, math, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sls_calib import CalibImage, Calibrator


# ====================================================================
# 标定板PnP：从图像计算相机位姿
# ====================================================================

class BoardPoseEstimator:
    """对标定板做PnP，输出每张图的相机位姿(C_T_G的反函数→G_T_C)。"""

    def __init__(self, K: np.ndarray, dist: np.ndarray,
                 board_yaml: str = "configs/board_points.yaml",
                 circle_interval: float = 25.0,
                 large_thresh: float = 0.55):
        self.K = K.astype(np.float64)
        self.dist = np.asarray(dist, dtype=np.float64).ravel()
        with open(board_yaml) as f:
            board = yaml.safe_load(f)
        self.pts3d = np.array(list(board['points'].values()), dtype=np.float64)
        self.pt_ids = list(board['points'].keys())
        self.interval = circle_interval
        self.thresh = large_thresh

    def process_image(self, img: np.ndarray
                      ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
                                 List[str], List[Tuple[float, float]], float]:
        """
        处理单张图: 检测标定板 → PnP → 相机位姿。
        Returns (G_R_C, G_t_C, point_ids, 2d_points, rmse)
        相机在G系中的位姿: X_G = G_R_C @ X_C + G_t_C
        """
        ci = CalibImage(name='tmp', image=img, selected=True)
        calib = Calibrator()
        calib.extract_circles([ci], only_selected=False, smooth=True, debug=False)
        ci.create_display_circles()
        ci.find_circle_indices(self.interval, debug=False,
                               large_circle_threshold=self.thresh)

        # 收集有效的2D-3D对应
        obj, img_pts, ids = [], [], []
        for (px,py),(wx,wy,wz),ok,_ in ci.circle_array:
            if ok:
                obj.append([wx,wy,wz])
                img_pts.append([px,py])
                ids.append(None)  # 不需要ID，只需要对应

        if len(obj) < 6:
            return None, None, [], [], 999

        obj_arr = np.array(obj, dtype=np.float64)
        img_arr = np.array(img_pts, dtype=np.float64)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_arr, img_arr, self.K, self.dist,
            flags=cv2.SOLVEPNP_EPNP, iterationsCount=200,
            reprojectionError=2.0, confidence=0.99)

        if not success:
            return None, None, [], [], 999

        # C_T_G: 标定板系→相机系
        C_R_G, _ = cv2.Rodrigues(rvec)
        C_t_G = tvec.ravel()

        # 求逆: G_T_C = inv(C_T_G)
        G_R_C = C_R_G.T
        G_t_C = -G_R_C @ C_t_G

        # 重投影RMSE
        proj, _ = cv2.projectPoints(obj_arr, rvec, tvec, self.K, self.dist)
        errs = np.linalg.norm(proj.reshape(-1,2) - img_arr, axis=1)
        rmse = float(np.sqrt(np.mean(errs**2)))

        return G_R_C, G_t_C, ids, img_pts, rmse


# ====================================================================
# 多点跟踪器
# ====================================================================

class PointTracker:
    """在图像序列中跟踪标志点。"""

    def __init__(self, win_size: Tuple[int, int] = (31, 31)):
        self.win_size = win_size
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                         30, 0.001)

    def track(self, prev_gray: np.ndarray, curr_gray: np.ndarray,
              prev_pts: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray]:
        """
        KLT光流法跟踪点。
        Returns (curr_pts, status) — status[i]=1表示跟踪成功。
        """
        if len(prev_pts) == 0:
            return np.zeros((0,2)), np.zeros(0)

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray,
            prev_pts.reshape(-1,1,2).astype(np.float32), None,
            winSize=self.win_size, criteria=self.criteria)
        return curr_pts.reshape(-1,2), status.ravel()


# ====================================================================
# 三角测量器
# ====================================================================

class MultiViewTriangulator:
    """从多视图观测三角测量三维点（相机位姿已知）。"""

    @staticmethod
    def triangulate(observations: List[Tuple[np.ndarray, np.ndarray,
                                              Tuple[float, float]]]
                    ) -> Optional[np.ndarray]:
        """
        Args:
            observations: [(G_R_C, G_t_C, (u,v)), ...] for each view.

        Returns:
            3D point in G frame, or None.
        """
        if len(observations) < 2:
            return None

        n = len(observations)
        A = np.zeros((2*n, 4), dtype=np.float64)

        for i, (G_R_C, G_t_C, (u, v)) in enumerate(observations):
            # P = K @ [R | t] where [R|t] maps world(G) → camera
            C_R_G = G_R_C.T
            C_t_G = -C_R_G @ G_t_C
            P = np.hstack([C_R_G, C_t_G.reshape(3,1)])
            A[2*i]     = u * P[2] - P[0]
            A[2*i + 1] = v * P[2] - P[1]

        _, _, Vt = np.linalg.svd(A, full_matrices=False)
        X = Vt[-1]
        p = X[:3] / X[3]

        # 检查在所有相机前方
        for G_R_C, G_t_C, _ in observations:
            C_R_G = G_R_C.T
            C_t_G = -C_R_G @ G_t_C
            pc = C_R_G @ p + C_t_G
            if pc[2] <= 0:
                return None

        return p

    @staticmethod
    def reprojection_error(p3d: np.ndarray, observations: List
                           ) -> Tuple[float, List[float]]:
        """计算一个3D点的重投影误差。"""
        errors = []
        for G_R_C, G_t_C, (u, v) in observations:
            C_R_G = G_R_C.T
            C_t_G = -C_R_G @ G_t_C
            pc = C_R_G @ p3d + C_t_G
            if pc[2] <= 0:
                errors.append(999)
                continue
            K = np.array([[1,0,0],[0,1,0],[0,0,1]])  # placeholder
            # Use the standard projection formula
            errors.append(float(np.linalg.norm([pc[0]/pc[2], pc[1]/pc[2]])))
        return np.sqrt(np.mean([e**2 for e in errors])), errors


# ====================================================================
# 主GUI
# ====================================================================

class AircraftTriangulationGUI:
    """标注 + 跟踪 + 三角测量的统一界面。"""

    def __init__(self, images: List[np.ndarray],
                 board_estimator: BoardPoseEstimator,
                 K: np.ndarray):
        self.images = images
        self.K = K.astype(np.float64)
        self.board_estimator = board_estimator
        self.tracker = PointTracker()
        self.triangulator = MultiViewTriangulator()

        # 标定板PnP结果（每张图）
        self.poses: List[Tuple] = []  # [(G_R_C, G_t_C, rmse), ...]
        self.valids: List[bool] = []

        # 飞机标志点（在第一个视图的2D位置）
        self.point_names: List[str] = []
        self.base_pts: np.ndarray = np.zeros((0, 2))  # 第一张图的标注

        # 每个点在所有视图的跟踪结果
        # observations[point_idx] = [(img_idx, u, v), ...]
        self.observations: Dict[int, List[Tuple[int, float, float]]] = {}

        # 三角测量结果
        self.points3d: Dict[str, np.ndarray] = {}
        self.errors: Dict[str, float] = {}

    def process_all_boards(self):
        """对所有图像做标定板PnP。"""
        print("--- 标定板PnP ---")
        for i, img in enumerate(self.images):
            G_R_C, G_t_C, ids, pts2d, rmse = self.board_estimator.process_image(img)
            ok = G_R_C is not None
            self.poses.append((G_R_C, G_t_C, rmse))
            self.valids.append(ok)
            status = f"RMSE={rmse:.2f}px" if ok else "FAIL"
            print(f"  图{i}: {status}")

        n_ok = sum(self.valids)
        print(f"  有效: {n_ok}/{len(self.images)}")

    def label_first_frame(self):
        """在第一张图上用鼠标标注点。"""
        print("\n--- 标注第一张图的飞机标志点 ---")
        print("  鼠标左键: 标注一个点")
        print("  按 'n': 命名上一个点（输入编号如 A001）")
        print("  按 'd': 删除上一个点")
        print("  按 's': 保存并继续")
        print("  按 'q': 退出")

        img = self.images[0].copy()
        pts = []
        names = []
        window = "标注 - 图0"

        def click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                pts.append((float(x), float(y)))
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(img, str(len(pts)), (x+10, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(window, img)

        cv2.namedWindow(window)
        cv2.setMouseCallback(window, click)
        cv2.imshow(window, img)

        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                return False
            elif key == ord('s') and len(pts) > 0:
                break
            elif key == ord('d') and len(pts) > 0:
                pts.pop()
                img = self.images[0].copy()
                for j, (px, py) in enumerate(pts):
                    cv2.circle(img, (int(px), int(py)), 5, (0, 255, 0), -1)
                    if j < len(names):
                        cv2.putText(img, names[j], (int(px)+10, int(py)-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow(window, img)
            elif key == ord('n') and len(pts) > 0:
                # 控制台输入名称
                name = input(f"  点{len(names)+1} 名称 (如A001): ").strip()
                if not name:
                    name = f"A{len(names)+1:03d}"
                names.append(name)
                cv2.putText(img, name, (int(pts[-1][0])+10, int(pts[-1][1])-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow(window, img)

        cv2.destroyAllWindows()

        # 确保所有点都有名字
        while len(names) < len(pts):
            name = input(f"  点{len(names)+1} 名称 (如A{len(names)+1:03d}): ").strip()
            if not name:
                name = f"A{len(names)+1:03d}"
            names.append(name)

        self.point_names = names
        self.base_pts = np.array(pts, dtype=np.float64)
        self.point_count = len(pts)

        # 初始化观测
        for i in range(self.point_count):
            self.observations[i] = [(0, pts[i][0], pts[i][1])]

        print(f"  已标注 {len(pts)} 个点: {names}")

        # 保存第一张图的可视化
        out = self.images[0].copy()
        for j, (px, py) in enumerate(pts):
            cv2.circle(out, (int(px), int(py)), 5, (0, 255, 0), -1)
            cv2.putText(out, names[j], (int(px)+10, int(py)-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite("output/aircraft_labels_frame0.png", out)
        print("  可视化: output/aircraft_labels_frame0.png")
        return True

    def track_all_frames(self):
        """用光流法跟踪所有点到后续帧。"""
        print("\n--- 跟踪 ---")
        prev_gray = cv2.cvtColor(self.images[0], cv2.COLOR_BGR2GRAY)
        prev_pts = self.base_pts.copy()

        for i in range(1, len(self.images)):
            if not self.valids[i]:
                continue

            curr_gray = cv2.cvtColor(self.images[i], cv2.COLOR_BGR2GRAY)
            curr_pts, status = self.tracker.track(prev_gray, curr_gray, prev_pts)

            tracked = 0
            for j in range(self.point_count):
                if status[j]:
                    pt = tuple(curr_pts[j])
                    self.observations[j].append((i, pt[0], pt[1]))
                    tracked += 1

            print(f"  图{i}: 跟踪 {tracked}/{self.point_count} 个点")
            prev_gray = curr_gray
            prev_pts = curr_pts

    def triangulate_all(self, K_actual: np.ndarray):
        """对所有点做多视图三角测量。"""
        print("\n--- 三角测量 ---")
        for j in range(self.point_count):
            obs_list = []
            for img_idx, u, v in self.observations[j]:
                if img_idx < len(self.poses) and self.valids[img_idx]:
                    G_R_C, G_t_C, _ = self.poses[img_idx]
                    obs_list.append((G_R_C, G_t_C, (u, v)))

            p3d = self.triangulator.triangulate(obs_list)
            if p3d is not None:
                name = self.point_names[j]
                self.points3d[name] = p3d
                # 计算重投影误差
                errs = []
                for G_R_C, G_t_C, (u, v) in obs_list:
                    C_R_G = G_R_C.T
                    C_t_G = -C_R_G @ G_t_C
                    pc = C_R_G @ p3d + C_t_G
                    if pc[2] > 0:
                        up = pc[0]/pc[2]*K_actual[0,0] + K_actual[0,2]
                        vp = pc[1]/pc[2]*K_actual[1,1] + K_actual[1,2]
                        errs.append(math.hypot(up-u, vp-v))
                self.errors[name] = float(np.mean(errs)) if errs else 0
                n_views = len(obs_list)
                print(f"  {name}: pos=({p3d[0]:.1f},{p3d[1]:.1f},{p3d[2]:.1f})mm, "
                      f"err={self.errors[name]:.2f}px, views={n_views}")
            else:
                print(f"  {self.point_names[j]}: 三角测量失败")

    def export_yaml(self, output_path: str):
        """输出为 aircraft_points.yaml。"""
        with open("configs/aircraft_points.yaml") as f:
            template = yaml.safe_load(f)

        template['notes'] = ('由多视图三角测量生成，'
                             f'{len(self.points3d)}个点，'
                             f'使用{sum(self.valids)}张有效图像')
        template['points'] = {}
        for name, p3d in self.points3d.items():
            region = 'unknown'
            if name in template.get('points', {}):
                region = template['points'][name].get('region', 'unknown')
            template['points'][name] = {
                'x_mm': float(p3d[0]),
                'y_mm': float(p3d[1]),
                'z_mm': float(p3d[2]),
                'region': region,
                'reproj_error_px': float(self.errors.get(name, 0)),
            }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(template, f, default_flow_style=False,
                      allow_unicode=True, sort_keys=False)
        print(f"\n飞机点三维坐标已导出: {output_path}")


# ====================================================================
# 主函数
# ====================================================================

def main():
    import argparse, os
    p = argparse.ArgumentParser(
        description='多视图三角测量飞机标志点三维坐标')
    p.add_argument('images', nargs='+', help='标定板+飞机同框的图像序列')
    p.add_argument('--config', default='configs/experiment_config.yaml')
    p.add_argument('--output', '-o', default='configs/aircraft_points_measured.yaml')
    args = p.parse_args()

    os.makedirs('output', exist_ok=True)

    # 读配置
    with open(args.config) as f: exp = yaml.safe_load(f)
    cal = exp['calibration']
    K = np.array([[cal['fx'], 0, cal['cx']],
                   [0, cal['fy'], cal['cy']],
                   [0, 0, 1]], dtype=np.float64)
    dist = np.array(cal['dist'], dtype=np.float64)

    # 读图像
    print(f"加载 {len(args.images)} 张图像...")
    images = []
    for pth in args.images:
        img = cv2.imread(pth)
        if img is None:
            print(f"  警告: 无法读取 {pth}")
            continue
        images.append(img)
    print(f"  成功加载 {len(images)} 张")

    if len(images) < 3:
        print("至少需要3张图像"); sys.exit(1)

    # 初始化
    board_est = BoardPoseEstimator(K, dist)
    gui = AircraftTriangulationGUI(images, board_est, K)

    # 步骤1: 所有图像跑标定板PnP
    gui.process_all_boards()
    if sum(gui.valids) < 3:
        print("有效图像不足"); sys.exit(1)

    # 步骤2: 标注第一张图
    if not gui.label_first_frame():
        print("标注取消"); sys.exit(0)

    # 步骤3: 跟踪
    gui.track_all_frames()

    # 步骤4: 三角测量
    gui.triangulate_all(K)

    # 步骤5: 输出
    if gui.points3d:
        gui.export_yaml(args.output)
        # 复制到标准位置
        import shutil
        shutil.copy(args.output, 'configs/aircraft_points.yaml')
        print("已更新 configs/aircraft_points.yaml")
    else:
        print("三角测量失败，无输出")


if __name__ == '__main__':
    main()
