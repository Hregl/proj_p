"""
多视图三角测量飞机标志点三维坐标 (v2)。

利用标定板PnP获得每张图的相机位姿，通过多视图三角测量直接解算
飞机标志点在标定板参考系(G)下的真实三维坐标（包括Z/高度）。

与v1的区别:
  - 使用 cv2.triangulatePoints 正确引入K矩阵
  - 跟踪后提供交互式修正界面（拖拽/增删观测点）
  - 重投影误差按视图分解，支持异常视图剔除
  - true inverse for rotation (np.linalg.inv)

使用方式:
  # 所有点在一张图中可见时:
  python tools/triangulate_aircraft_points.py data/tri/*.png \
      --point-names 机舱顶 左翼尖 右翼尖 机脊中部 \
                    左横尾翼尖 右横尾翼尖 左竖尾翼尖 右竖尾翼尖

  # 部分点在某帧不可见时(同样的命令, 标注时挑可见的点标即可):
  python tools/triangulate_aircraft_points.py data/tri/*.png --threshold 0.55

流程:
  1. 每张图检测标定板 → PnP → 相机位姿 (G_R_C, G_t_C)
  2. 注册所有标志点名称 (--point-names 或交互输入)
  3. 图0标注: 点击位置→按数字键分配点号 → s保存
  4. 后续帧逐帧标注: 橙色参考圈=上帧位置, 点击→按数字键 → s保存
  5. cv2.triangulatePoints 三角测量 → 3D坐标 (带真实Z)
  6. 异常视图自动剔除 & 重三角测量
  7. 重投影误差分析 → 导出 aircraft_points_measured.yaml
"""

import sys, yaml, cv2, numpy as np, math, os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sls_calib import CalibImage, Calibrator


# ====================================================================
# 标定板PnP
# ====================================================================

class BoardPoseEstimator:
    """对每张图做标定板PnP，输出相机在G系中的位姿。"""

    def __init__(self, K: np.ndarray, dist: np.ndarray,
                 board_yaml: str = "configs/board_points.yaml",
                 circle_interval: float = 25.0,
                 large_thresh: float = 0.55,
                 use_blob: bool = True):
        self.K = K.astype(np.float64)
        self.dist = np.asarray(dist, dtype=np.float64).ravel()
        with open(board_yaml, encoding='utf-8') as f:
            board = yaml.safe_load(f)
        self.pts3d = np.array(list(board['points'].values()), dtype=np.float64)
        self.pt_ids = list(board['points'].keys())
        self.interval = circle_interval
        self.thresh = large_thresh
        self.use_blob = use_blob

    def process_image(self, img: np.ndarray
                      ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
                                 float, np.ndarray, np.ndarray]:
        """
        Returns:
            G_R_C, G_t_C, rmse, rvec, tvec
            相机在G系中的位姿: X_G = G_R_C @ X_C + G_t_C
            rvec/tvec: 原始PnP结果 (C_T_G), 用于cv2.projectPoints
        """
        ci = CalibImage(name='tmp', image=img, selected=True)

        if self.use_blob:
            from sls_calib.board_detector import BoardDetector
            bd = BoardDetector()
            markers, _ = bd.detect(img)
            # Convert to CalibImage.circles format: [((cx,cy), area), ...]
            ci.circles = [((cx, cy), area) for (cx, cy), area in markers]
            ci.create_display_circles()
        else:
            calib = Calibrator()
            calib.extract_circles([ci], only_selected=False, smooth=True, debug=False)
            ci.create_display_circles()

        # Auto-tune: try thresholds, pick best (high assigned, low RMSE)
        best_score, best_t = -1, self.thresh
        best_circle_array = None
        thresholds = [self.thresh] if self.thresh else \
                     [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
        for t in thresholds:
            ci2 = CalibImage(name='tmp', image=img, selected=True)
            ci2.circles = ci.circles
            ci2.display_circles = ci.display_circles
            err = ci2.find_circle_indices(self.interval, debug=False,
                                          large_circle_threshold=t)
            if err != '':
                continue
            n_assigned = sum(1 for _, _, ok, _ in ci2.circle_array if ok)
            if n_assigned < 6:
                continue
            # Quick PnP to validate this threshold
            obj_t, img_t = [], []
            for (px, py), (wx, wy, wz), ok, _ in ci2.circle_array:
                if ok:
                    obj_t.append([wx, wy, wz])
                    img_t.append([px, py])
            obj_t = np.array(obj_t, dtype=np.float64)
            img_t = np.array(img_t, dtype=np.float64)
            ok_pnp, rv, tv = cv2.solvePnP(
                obj_t, img_t, self.K, self.dist,
                flags=cv2.SOLVEPNP_IPPE)
            if not ok_pnp:
                continue
            # Project all points and compute RMSE
            proj_t, _ = cv2.projectPoints(obj_t, rv, tv, self.K, self.dist)
            errs_t = np.linalg.norm(proj_t.reshape(-1, 2) - img_t, axis=1)
            rmse_t = float(np.sqrt(np.mean(errs_t**2)))
            if rmse_t > 10.0:  # Reject clearly wrong assignments
                continue
            # Score: prefer more assigned circles, penalize high RMSE
            score = n_assigned - rmse_t * 5
            if score > best_score:
                best_score = score
                best_t = t
                best_circle_array = ci2.circle_array

        if best_circle_array is None:
            return None, None, 999, None, None
        ci.circle_array = best_circle_array

        obj, img_pts = [], []
        for (px, py), (wx, wy, wz), ok, _ in ci.circle_array:
            if ok:
                obj.append([wx, wy, wz])
                img_pts.append([px, py])

        obj_arr = np.array(obj, dtype=np.float64)
        img_arr = np.array(img_pts, dtype=np.float64)

        # IPPE: designed for planar objects (Z=0 board). EPNP fails with all-coplanar points.
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_arr, img_arr, self.K, self.dist,
            flags=cv2.SOLVEPNP_IPPE, iterationsCount=200,
            reprojectionError=2.0, confidence=0.99)
        if not success:
            # Fallback: IPPE without RANSAC
            success, rvec, tvec = cv2.solvePnP(
                obj_arr, img_arr, self.K, self.dist,
                flags=cv2.SOLVEPNP_IPPE)
            inliers = None
        if not success:
            return None, None, 999, None, None

        # C_T_G: 标定板系→相机系
        C_R_G, _ = cv2.Rodrigues(rvec)
        C_t_G = tvec.ravel()

        # 求逆: G_T_C = inv(C_T_G)  (使用真逆)
        G_R_C = np.linalg.inv(C_R_G)
        G_t_C = -G_R_C @ C_t_G

        # 重投影RMSE (仅内点)
        if inliers is not None and len(inliers) > 0:
            inl_idx = inliers.ravel()
            mask = np.zeros(len(obj_arr), dtype=bool)
            mask[inl_idx] = True
            proj, _ = cv2.projectPoints(obj_arr[mask], rvec, tvec, self.K, self.dist)
            errs = np.linalg.norm(proj.reshape(-1, 2) - img_arr[mask], axis=1)
        else:
            proj, _ = cv2.projectPoints(obj_arr, rvec, tvec, self.K, self.dist)
            errs = np.linalg.norm(proj.reshape(-1, 2) - img_arr, axis=1)
        rmse = float(np.sqrt(np.mean(errs**2)))

        return G_R_C, G_t_C, rmse, rvec, tvec


# ====================================================================
# KLT 跟踪器
# ====================================================================

class PointTracker:
    """KLT光流法在图像序列中跟踪标志点。"""

    def __init__(self, win_size: Tuple[int, int] = (31, 31)):
        self.win_size = win_size
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                         30, 0.001)

    def track(self, prev_gray: np.ndarray, curr_gray: np.ndarray,
              prev_pts: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (curr_pts, status). status[i]=1 if tracked successfully."""
        if len(prev_pts) == 0:
            return np.zeros((0, 2)), np.zeros(0, dtype=np.uint8)

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray,
            prev_pts.reshape(-1, 1, 2).astype(np.float32), None,
            winSize=self.win_size, criteria=self.criteria)
        return curr_pts.reshape(-1, 2), status.ravel()


# ====================================================================
# 主GUI: 标注 + 跟踪修正 + 三角测量
# ====================================================================

class AircraftTriangulationGUI:
    """标注、跟踪修正、三角测量的统一界面。"""

    MAX_DISPLAY_W = 1400  # 显示窗口最大宽度
    MAX_DISPLAY_H = 900   # 显示窗口最大高度

    def __init__(self, images: List[np.ndarray],
                 image_paths: List[str],
                 board_estimator: BoardPoseEstimator,
                 K: np.ndarray, dist: np.ndarray):
        self.images = images
        self.image_paths = image_paths
        self.K = K.astype(np.float64)
        self.dist = np.asarray(dist, dtype=np.float64).ravel()
        self.board_estimator = board_estimator
        self.tracker = PointTracker()

        # 标定板PnP结果
        self.poses: List[Dict] = []  # [{G_R_C, G_t_C, rmse, rvec, tvec}, ...]
        self.valids: List[bool] = []

        # 标志点 (全部预注册，包括可能在某些帧不可见的)
        self.point_names: List[str] = []
        self.point_count: int = 0
        self.point_descriptions: Dict[str, str] = {}

        # 观测: observations[point_idx] = [(img_idx, u, v), ...]
        self.observations: Dict[int, List[Tuple[int, float, float]]] = {}

        # 三角测量结果
        self.points3d: Dict[str, np.ndarray] = {}
        self.errors: Dict[str, Dict] = {}  # {name: {mean, per_view: [...]}}

        # 每帧的显示缩放因子 (在 _resize_img 中计算)
        self._scales: Dict[int, float] = {}

    def _resize_img(self, img_idx: int) -> Tuple[np.ndarray, float]:
        """缩放图像以适应显示窗口。返回 (resized_image, scale)。"""
        img = self.images[img_idx]
        h, w = img.shape[:2]
        scale = min(self.MAX_DISPLAY_W / w, self.MAX_DISPLAY_H / h, 1.0)
        if scale < 1.0:
            resized = cv2.resize(img, (int(w * scale), int(h * scale)))
        else:
            resized = img.copy()
            scale = 1.0
        self._scales[img_idx] = scale
        return resized, scale

    # ------------------------------------------------------------------
    def process_all_boards(self):
        """对所有图像做标定板PnP。"""
        print("\n" + "="*60)
        print("Step 1: Board PnP (solving camera pose for each image)")
        print("="*60)
        for i, img in enumerate(self.images):
            G_R_C, G_t_C, rmse, rvec, tvec = self.board_estimator.process_image(img)
            ok = G_R_C is not None
            self.poses.append({
                'G_R_C': G_R_C, 'G_t_C': G_t_C, 'rmse': rmse,
                'rvec': rvec, 'tvec': tvec
            })
            self.valids.append(ok)
            status = f"RMSE={rmse:.2f}px" if ok else "FAIL"
            # Mark frames with insane RMSE as invalid (wrong large circle identification)
            if ok and rmse > 50.0:
                ok = False
                self.valids[i] = False
                status = f"REJECTED (RMSE={rmse:.1f}px > 50, wrong large circles)"
            name = Path(self.image_paths[i]).name if i < len(self.image_paths) else f"img{i}"
            print(f"  [{i}] {name}: {status}")

        n_ok = sum(self.valids)
        print(f"  Valid: {n_ok}/{len(self.images)}")
        if n_ok < 3:
            print("  WARNING: Valid images < 3, triangulation may fail")

    # ------------------------------------------------------------------
    def register_point_names(self, preset_names: Optional[List[str]] = None):
        """预注册所有标志点名称(包括某些帧不可见的)。"""
        print("\n" + "="*60)
        print("Step 2: Register point names")
        print("="*60)

        if preset_names:
            self.point_names = list(preset_names)
            print(f"  Using preset names: {self.point_names}")
        else:
            # 交互式输入
            while True:
                try:
                    n = int(input("  How many marker points? ").strip())
                    if n >= 3:
                        break
                    print("  Need at least 3 points")
                except ValueError:
                    pass

            print(f"  Enter names for {n} points in order (e.g. nose tip left_wing right_wing ...)")
            for i in range(n):
                name = input(f"  Point {i+1}/{n}: ").strip()
                if not name:
                    name = f"B{i+1:03d}"
                self.point_names.append(name)

        self.point_count = len(self.point_names)

        # 初始化空观测
        for i in range(self.point_count):
            self.observations[i] = []

        print(f"  Registered {self.point_count} points: {self.point_names}")
        print(f"  Note: not all points need to be visible in every image; each point only needs >=3 views")

    # ------------------------------------------------------------------
    def _annotate_one_frame(self, img_idx: int,
                            hints: Optional[Dict[int, Tuple[float, float]]] = None
                            ) -> Optional[Dict[int, Tuple[float, float]]]:
        """统一的单帧标注GUI: 点击→按数字键分配。返回 {pt_idx: (u,v)} 或 None=退出。

        hints: 可选的上帧标注位置(以原始像素坐标表示), 显示为淡色参考圆。
        """
        frame_obs: Dict[int, Tuple[float, float]] = {}
        display_img, scale = self._resize_img(img_idx)
        dh, dw = display_img.shape[:2]

        fname = Path(self.image_paths[img_idx]).name
        window = f"Frame {img_idx} - {fname} | Click -> press 1-8 -> s:save"

        def to_display(px, py):
            return int(px * scale), int(py * scale)

        def redraw():
            nonlocal display_img
            display_img, _ = self._resize_img(img_idx)
            # 画参考位置 (上帧标注)
            if hints:
                for pt_idx, (hx, hy) in hints.items():
                    dx, dy = to_display(hx, hy)
                    cv2.circle(display_img, (dx, dy), 12, (255, 150, 50), 1)
                    cv2.putText(display_img, f"[{pt_idx+1}]", (dx+14, dy-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 150, 50), 1)
            # 画已标注的点
            for pt_idx, (px, py) in sorted(frame_obs.items()):
                dx, dy = to_display(px, py)
                cv2.circle(display_img, (dx, dy), 7, (0, 255, 0), -1)
                cv2.circle(display_img, (dx, dy), 9, (0, 255, 0), 2)
                cv2.putText(display_img, f"[{pt_idx+1}]", (dx+14, dy-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            # 状态栏
            parts = []
            for i in range(self.point_count):
                m = "v" if i in frame_obs else "x"
                parts.append(f"[{i+1}]{m}")
            cv2.putText(display_img, " | ".join(parts),
                       (5, dh-8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            cv2.putText(display_img,
                       "Click -> press 1-8 -> s:save  q:quit",
                       (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        redraw()
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, dw, dh)
        cv2.imshow(window, display_img)

        pending: Optional[Tuple[float, float]] = None  # display coords

        def click(event, x, y, flags, param):
            nonlocal pending
            if event == cv2.EVENT_LBUTTONDOWN:
                pending = (float(x), float(y))
                tmp = display_img.copy()
                cv2.circle(tmp, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(tmp, "Press 1-8", (x+10, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                cv2.imshow(window, tmp)

        cv2.setMouseCallback(window, click)

        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                return None
            elif key == ord('s'):
                if len(frame_obs) > 0:
                    break
            elif key == ord('d') and pending is not None:
                pending = None
                redraw()
                cv2.imshow(window, display_img)
            elif ord('1') <= key <= ord('9'):
                idx = key - ord('1')
                if idx < self.point_count and pending is not None:
                    # Convert display coords to original pixel coords
                    orig_x = pending[0] / scale
                    orig_y = pending[1] / scale

                    # Sub-pixel refinement
                    from sls_calib.point_refiner import PointRefiner
                    refiner = PointRefiner()
                    refined = refiner.refine(
                        self.images[img_idx], orig_x, orig_y, debug=False)

                    if refined and refined['offset_px'] < 30:
                        use_x, use_y = refined['pixel_x'], refined['pixel_y']
                        conf = refined['confidence']
                        off = refined['offset_px']
                        print(f"    [{idx+1}] {self.point_names[idx]}: "
                              f"refined ({use_x:.1f}, {use_y:.1f}) "
                              f"offset={off:.1f}px conf={conf:.2f}")
                    else:
                        use_x, use_y = orig_x, orig_y
                        print(f"    [{idx+1}] {self.point_names[idx]}: "
                              f"raw click ({orig_x:.0f}, {orig_y:.0f}) "
                              f"(refinement failed)")

                    frame_obs[idx] = (use_x, use_y)
                    pending = None
                    redraw()
                    cv2.imshow(window, display_img)

        cv2.destroyAllWindows()
        return frame_obs

    def label_first_frame(self, existing_labels: Optional[str] = None):
        """标注第一张图 (所有可见的标志点)。"""
        print("\n" + "="*60)
        print("Step 3: Label marker points (frame by frame)")
        print("="*60)
        print(f"  Point index mapping: " + ", ".join(
            f"[{i+1}]={name}" for i, name in enumerate(self.point_names)))
        print(f"  Per frame: click position -> press number key to assign point -> s to save")

        # 加载已有标注
        hints = None
        if existing_labels and os.path.exists(existing_labels):
            print(f"  Loading existing labels: {existing_labels}")
            with open(existing_labels, encoding='utf-8') as f:
                data = yaml.safe_load(f)
            hints = {}
            name_to_idx = {n: i for i, n in enumerate(self.point_names)}
            for name, info in data.get('points', {}).items():
                if name in name_to_idx:
                    px, py = float(info['pixel_x']), float(info['pixel_y'])
                    if px >= 0 and py >= 0:
                        hints[name_to_idx[name]] = (px, py)
            print(f"  Loaded {len(hints)} reference positions")

        # 标注帧0
        print(f"\n  --- Frame 0: {Path(self.image_paths[0]).name} ---")
        frame_obs = self._annotate_one_frame(0, hints=hints)
        if frame_obs is None:
            return False

        for idx, (px, py) in frame_obs.items():
            self.observations[idx].append((0, px, py))

        n = len(frame_obs)
        print(f"  Frame 0: labeled {n}/{self.point_count} points"
              + (f", {self.point_count-n} not visible (fill in later frames)" if n < self.point_count else ""))

        self._save_frame_labels(0, [frame_obs.get(i, (-1, -1)) for i in range(self.point_count)])
        return True

    # ------------------------------------------------------------------
    def _save_frame_labels(self, img_idx: int, pts_2d: List[Tuple[float, float]]):
        """Save per-frame annotations as YAML with refinement metadata."""
        img_name = Path(self.image_paths[img_idx]).stem
        out_dir = Path("annotations/aircraft_2d")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{img_name}_points.yaml"

        data = {
            'image': Path(self.image_paths[img_idx]).name,
            'point_count': sum(1 for px, py in pts_2d if px >= 0),
            'points': {}
        }
        for j, (px, py) in enumerate(pts_2d):
            name = self.point_names[j] if j < len(self.point_names) else f"P{j+1}"
            visible = px >= 0
            data['points'][name] = {
                'pixel_x': float(px),
                'pixel_y': float(py),
                'source': 'subpixel_refined',  # PointRefiner applied at click time
                'visible': visible,
            }
        data['source'] = 'subpixel_refined'

        with open(out_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # ------------------------------------------------------------------
    def track_all_frames(self):
        """KLT光流法跟踪。只跟踪上一帧有观测的点，无观测的点跳过。"""
        print("\n" + "="*60)
        print("Step 4: KLT optical flow tracking")
        print("="*60)

        # 第一帧: 收集有效观测点
        prev_frame_mask = [False] * self.point_count  # 哪些点在上一帧有观测
        prev_pts_list = []
        prev_pt_indices = []  # prev_pts[j] 对应 self.point_names[prev_pt_indices[j]]
        for i in range(self.point_count):
            for obs in self.observations[i]:
                if obs[0] == 0:
                    prev_pts_list.append([obs[1], obs[2]])
                    prev_pt_indices.append(i)
                    prev_frame_mask[i] = True
                    break

        if not prev_pts_list:
            print("  No valid observations in first frame, cannot track")
            return

        prev_pts = np.array(prev_pts_list, dtype=np.float32)
        prev_gray = cv2.cvtColor(self.images[0], cv2.COLOR_BGR2GRAY)

        for img_i in range(1, len(self.images)):
            # KLT跟踪始终进行(保持连续性), 但仅对PnP有效的帧保存观测
            curr_gray = cv2.cvtColor(self.images[img_i], cv2.COLOR_BGR2GRAY)

            if len(prev_pts) == 0:
                print(f"  [{img_i}] No points to track, skipping")
                continue

            curr_pts, status = self.tracker.track(prev_gray, curr_gray, prev_pts)

            tracked, lost, saved = 0, 0, 0
            new_prev_pts = []
            new_prev_indices = []

            for j in range(len(prev_pt_indices)):
                pt_idx = prev_pt_indices[j]
                if j < len(status) and status[j]:
                    u, v = float(curr_pts[j][0]), float(curr_pts[j][1])
                    # 仅PnP有效的帧才保存观测(用于三角测量)
                    if self.valids[img_i]:
                        self.observations[pt_idx].append((img_i, u, v))
                        saved += 1
                    new_prev_pts.append([u, v])
                    new_prev_indices.append(pt_idx)
                    tracked += 1
                else:
                    lost += 1

            name = Path(self.image_paths[img_i]).name if img_i < len(self.image_paths) else f"img{img_i}"
            pnp_ok = "OK" if self.valids[img_i] else "NO_PNP"
            print(f"  [{img_i}] {name}: tracked {tracked} lost {lost} saved {saved} | PnP={pnp_ok}")

            prev_pts = np.array(new_prev_pts, dtype=np.float32) if new_prev_pts else np.zeros((0, 2))
            prev_pt_indices = new_prev_indices
            prev_gray = curr_gray

    # ------------------------------------------------------------------
    def annotate_remaining_frames(self):
        """对每个PnP有效的后续帧独立标注。跳过KLT, 上一帧标注作为橙色参考。"""
        print("\n" + "="*60)
        print("Step 4: Label remaining frames (frame by frame)")
        print("="*60)
        print(f"  Point index mapping: " + ", ".join(
            f"[{i+1}]={name}" for i, name in enumerate(self.point_names)))
        print(f"  Orange dashed circle = previous frame position (reference)  Green solid = current frame labeled")

        prev_hints: Dict[int, Tuple[float, float]] = {}
        for i in range(self.point_count):
            if self.observations[i]:
                last = self.observations[i][-1]
                prev_hints[i] = (last[1], last[2])

        for img_i in range(1, len(self.images)):
            if not self.valids[img_i]:
                continue

            fname = Path(self.image_paths[img_i]).name
            rmse = self.poses[img_i]['rmse']
            print(f"\n  --- Frame {img_i}: {fname} (PnP RMSE={rmse:.2f}px) ---")

            frame_obs = self._annotate_one_frame(img_i, hints=prev_hints)
            if frame_obs is None:
                print("    Skipping remaining frames")
                break

            for pt_idx, (px, py) in frame_obs.items():
                self.observations[pt_idx].append((img_i, px, py))

            n = len(frame_obs)
            print(f"    Labeled {n}/{self.point_count} points")

            for pt_idx, (px, py) in frame_obs.items():
                prev_hints[pt_idx] = (px, py)

            self._save_frame_labels(
                img_i,
                [frame_obs.get(i, (-1, -1)) for i in range(self.point_count)]
            )

    # ------------------------------------------------------------------
    def triangulate_all(self):
        """使用 cv2.triangulatePoints 进行多视图三角测量。"""
        print("\n" + "="*60)
        print("Step 5: Triangulation")
        print("="*60)

        for j in range(self.point_count):
            name = self.point_names[j]
            obs_list = self.observations[j]

            if len(obs_list) < 2:
                print(f"  {name}: insufficient observations ({len(obs_list)} views), skipping")
                continue

            # Undistort 2D points to normalized coordinates, then triangulate
            # with P = [R|t] (no K). This correctly handles lens distortion
            # unlike the old K[R|t] approach which ignored dist.
            P_norm_list = []  # 3x4 projection [R|t] in normalized camera coords
            pts_norm_list = []  # undistorted normalized 2D points (x/z, y/z)
            view_indices = []

            for img_idx, u, v in obs_list:
                if img_idx < len(self.poses) and self.valids[img_idx]:
                    pose = self.poses[img_idx]
                    G_R_C = pose['G_R_C']
                    G_t_C = pose['G_t_C']

                    # C_T_G = inv(G_T_C): world(G)→camera
                    C_R_G = np.linalg.inv(G_R_C)
                    C_t_G = -C_R_G @ G_t_C

                    # Projection without K: normalized coords
                    P_norm_list.append(np.hstack([C_R_G, C_t_G.reshape(3, 1)]))
                    # Undistort: pixel → normalized (ideal) coords
                    pts_undist = cv2.undistortPoints(
                        np.array([[[u, v]]], dtype=np.float32),
                        self.K, self.dist, P=None)
                    pts_norm_list.append(pts_undist[0, 0].tolist())
                    view_indices.append(img_idx)

            if len(P_norm_list) < 2:
                print(f"  {name}: insufficient valid projection matrices, skipping")
                continue

            # Triangulation in normalized coordinates
            if len(P_norm_list) == 2:
                pts_a = np.array([[pts_norm_list[0][0]], [pts_norm_list[0][1]]],
                                 dtype=np.float32)
                pts_b = np.array([[pts_norm_list[1][0]], [pts_norm_list[1][1]]],
                                 dtype=np.float32)
                pts4d = cv2.triangulatePoints(
                    P_norm_list[0], P_norm_list[1], pts_a, pts_b)
            else:
                tri_pts = []
                for a in range(len(P_norm_list)):
                    for b in range(a+1, len(P_norm_list)):
                        pts_a = np.array([[pts_norm_list[a][0]],
                                          [pts_norm_list[a][1]]], dtype=np.float32)
                        pts_b = np.array([[pts_norm_list[b][0]],
                                          [pts_norm_list[b][1]]], dtype=np.float32)
                        try:
                            p4d = cv2.triangulatePoints(
                                P_norm_list[a], P_norm_list[b], pts_a, pts_b)
                            p3d_pair = p4d[:3, 0] / p4d[3, 0]
                            tri_pts.append(p3d_pair)
                        except cv2.error:
                            pass

                if not tri_pts:
                    print(f"  {name}: all view pairs failed triangulation")
                    continue

                tri_arr = np.array(tri_pts)
                median = np.median(tri_arr, axis=0)
                pts4d = np.array([[median[0]], [median[1]], [median[2]], [1.0]])

            p3d = pts4d[:3, 0] / pts4d[3, 0]

            # Check in front of all cameras
            all_front = True
            for img_idx, u, v in obs_list:
                if img_idx < len(self.poses) and self.valids[img_idx]:
                    pose = self.poses[img_idx]
                    G_R_C = pose['G_R_C']
                    G_t_C = pose['G_t_C']
                    C_R_G = np.linalg.inv(G_R_C)
                    C_t_G = -C_R_G @ G_t_C
                    pc = C_R_G @ p3d + C_t_G
                    if pc[2] <= 0:
                        all_front = False
                        break
            if not all_front:
                print(f"  {name}: triangulated point behind camera, skipping")
                continue

            # Reprojection error in DISTORTED pixel space
            per_view_errs = []
            for vi, (img_idx, u, v) in enumerate(
                    [(vi, u, v) for vi, u, v in obs_list
                     if vi in view_indices]):
                if img_idx < len(self.poses) and self.valids[img_idx]:
                    pose = self.poses[img_idx]
                    C_R_G = np.linalg.inv(pose['G_R_C'])
                    C_t_G = -C_R_G @ pose['G_t_C']
                    rvec, _ = cv2.Rodrigues(C_R_G)
                    proj, _ = cv2.projectPoints(
                        p3d.reshape(1, 3), rvec, C_t_G, self.K, self.dist)
                    u_p, v_p = proj[0, 0]
                    err = math.hypot(u_p - u, v_p - v)
                    per_view_errs.append((img_idx, err))

            mean_err = float(np.mean([e for _, e in per_view_errs]))

            self.points3d[name] = p3d
            self.errors[name] = {
                'mean': mean_err,
                'per_view': per_view_errs,
                'n_views': len(P_list),
            }

            # 打印结果
            view_str = ", ".join([f"v{v}({e:.1f}px)" for v, e in per_view_errs])
            print(f"  {name}: ({p3d[0]:.1f}, {p3d[1]:.1f}, {p3d[2]:.1f}) mm, "
                  f"err={mean_err:.2f}px, views={len(P_list)} | {view_str}")

    # ------------------------------------------------------------------
    def remove_outlier_views(self, max_error_px: float = 10.0):
        """剔除重投影误差过大的观测，重新三角测量。"""
        print("\n" + "="*60)
        print("Step 6: Outlier view removal & re-triangulation")
        print("="*60)

        removed = 0
        for j in range(self.point_count):
            name = self.point_names[j]
            if name not in self.errors:
                continue

            per_view = self.errors[name].get('per_view', [])
            bad_views = [v for v, e in per_view if e > max_error_px]

            if bad_views:
                # 剔除异常视图
                self.observations[j] = [
                    obs for obs in self.observations[j]
                    if obs[0] not in bad_views
                ]
                print(f"  {name}: removed views {bad_views} (error > {max_error_px}px), "
                      f"remaining {len(self.observations[j])} views")
                removed += len(bad_views)

        if removed > 0:
            print(f"\n  Removed {removed} outlier observations, re-triangulating...")
            self.points3d.clear()
            self.errors.clear()
            self.triangulate_all()
        else:
            print("  No outlier views to remove")

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _ba_refine_points(self):
        """Nonlinear refinement (bundle adjustment) of each 3D point.

        Minimizes reprojection error across all views using Huber loss.
        Also computes triangulation angle and quality metrics.
        """
        print("\n" + "="*60)
        print("Step 6b: Nonlinear point refinement (Huber BA)")
        print("="*60)

        try:
            from scipy.optimize import least_squares
        except ImportError:
            print("  scipy not available, skipping BA refinement")
            return

        for name, p3d in self.points3d.items():
            # Collect observations for this point
            obs = []
            for pt_idx, name_check in enumerate(self.point_names):
                if name_check != name:
                    continue
                for img_idx, u, v in self.observations.get(pt_idx, []):
                    if img_idx < len(self.poses) and self.valids[img_idx]:
                        obs.append((img_idx, u, v, self.poses[img_idx]))
                break

            if len(obs) < 2:
                continue

            def cost_func(p):
                """Reprojection error in distorted pixel space (uses full K+dist)."""
                residuals = []
                p3d_arr = p.reshape(1, 3).astype(np.float64)
                for img_idx, u, v, pose in obs:
                    G_R_C = pose['G_R_C']
                    G_t_C = pose['G_t_C']
                    C_R_G = np.linalg.inv(G_R_C)
                    C_t_G = -C_R_G @ G_t_C
                    # Check in front of camera
                    pc = C_R_G @ p + C_t_G
                    if pc[2] <= 0:
                        residuals.extend([999.0, 999.0])
                        continue
                    rvec, _ = cv2.Rodrigues(C_R_G)
                    proj, _ = cv2.projectPoints(
                        p3d_arr, rvec, C_t_G, self.K, self.dist)
                    u_p, v_p = proj[0, 0]
                    residuals.append(u_p - u)
                    residuals.append(v_p - v)
                return np.array(residuals)

            try:
                result = least_squares(cost_func, p3d, loss='huber',
                                       method='trf', max_nfev=50)
                if result.success:
                    p3d_refined = result.x
                    # Update 3D point
                    self.points3d[name] = p3d_refined

                    # Recompute errors using FULL camera model (K + dist)
                    per_view_errs = []
                    view_dirs = []
                    p3d_arr = p3d_refined.reshape(1, 3).astype(np.float64)
                    for img_idx, u, v, pose in obs:
                        G_R_C = pose['G_R_C']
                        G_t_C = pose['G_t_C']
                        C_R_G = np.linalg.inv(G_R_C)
                        C_t_G = -C_R_G @ G_t_C
                        pc = C_R_G @ p3d_refined + C_t_G
                        if pc[2] > 0:
                            rvec, _ = cv2.Rodrigues(C_R_G)
                            proj, _ = cv2.projectPoints(
                                p3d_arr, rvec, C_t_G, self.K, self.dist)
                            u_p, v_p = proj[0, 0]
                            err = math.hypot(u_p - u, v_p - v)
                            per_view_errs.append((img_idx, err))
                            cam_pt_dir = p3d_refined - G_t_C
                            view_dirs.append(cam_pt_dir / np.linalg.norm(cam_pt_dir))

                    mean_err = float(np.mean([e for _, e in per_view_errs]))
                    max_err = float(max([e for _, e in per_view_errs]))

                    # Triangulation angle: max angle between any two view directions
                    tri_angle = 0.0
                    for a in range(len(view_dirs)):
                        for b in range(a + 1, len(view_dirs)):
                            cos_ang = np.dot(view_dirs[a], view_dirs[b])
                            cos_ang = max(-1.0, min(1.0, cos_ang))
                            ang = math.degrees(math.acos(cos_ang))
                            if ang > tri_angle:
                                tri_angle = ang

                    # Quality grade
                    if mean_err <= 1.0 and max_err <= 2.0 and len(obs) >= 5 and tri_angle >= 8:
                        quality = 'good'
                    elif mean_err <= 2.0 and max_err <= 5.0 and len(obs) >= 3:
                        quality = 'fair'
                    else:
                        quality = 'poor'

                    self.errors[name] = {
                        'mean': mean_err,
                        'max': max_err,
                        'per_view': per_view_errs,
                        'n_views': len(obs),
                        'tri_angle_deg': round(tri_angle, 1),
                        'quality': quality,
                    }

                    nv = len(obs)
                    print(f"  {name}: err={mean_err:.2f} px (max={max_err:.1f}), "
                          f"angle={tri_angle:.1f} deg, views={nv}, quality={quality}")
                else:
                    print(f"  {name}: BA did not converge")

            except Exception as e:
                print(f"  {name}: BA failed ({e})")

    # ------------------------------------------------------------------
    def _rigid_body_check(self):
        """Check rigid-body consistency of the reconstructed points."""
        print("\n" + "="*60)
        print("Step 6c: Rigid-body geometry check")
        print("="*60)

        # Compute pairwise distances
        names = sorted(self.points3d.keys())
        if len(names) < 3:
            print("  Not enough points for rigid-body check")
            return {}

        # Pairwise distance matrix
        distances = {}
        for i, na in enumerate(names):
            for j, nb in enumerate(names):
                if i < j:
                    d = float(np.linalg.norm(self.points3d[na] - self.points3d[nb]))
                    distances[f'{na}-{nb}'] = round(d, 1)

        print(f"\n  Point-to-point distances (mm):")
        for pair in sorted(distances.keys()):
            d = distances[pair]
            flag = ' *** TOO CLOSE' if d < 5.0 else ''
            print(f"    {pair}: {d:.1f} mm{flag}")

        # Symmetry check: find pairs that should be symmetric
        # (e.g., left/right wingtip should be ~same distance from centerline)
        symmetry_pairs = [
            ('左翼尖', '右翼尖'),
            ('左横尾翼尖', '右横尾翼尖'),
            ('左竖尾翼尖', '右竖尾翼尖'),
        ]
        for left, right in symmetry_pairs:
            if left in self.points3d and right in self.points3d:
                pl = self.points3d[left]
                pr = self.points3d[right]
                # Check Y symmetry: |Y_left + Y_right| should be ≈ 0
                y_center = abs(float(pl[1] + pr[1]))
                # X should be similar for symmetric points
                x_diff = abs(float(pl[0] - pr[0]))
                # Distance from each to the centerline should be equal
                dl = abs(float(pl[1]))
                dr = abs(float(pr[1]))
                asym = abs(dl - dr)
                print(f"    {left} <-> {right}: Y_center_offset={y_center:.1f}mm, "
                      f"X_diff={x_diff:.1f}mm, L/R_asym={asym:.1f}mm")

        return distances

    def export_yaml(self, output_path: str):
        """导出为 aircraft_points.yaml (v3 with quality metrics)."""
        print("\n" + "="*60)
        print("Step 7: Export results")
        print("="*60)

        data = {
            'aircraft_name': 'model_jet',
            'coordinate_system': 'G',
            'unit': 'mm',
            'method': 'Multi-view triangulation + Huber BA refinement',
            'origin': 'Board origin',
            'point_count': len(self.points3d),
            'note': (f'Generated from {sum(self.valids)} valid images. '
                     f'Refined with Huber-loss nonlinear least squares. '
                     f'Z not forced to zero.'),
            'points': {}
        }

        for name, p3d in self.points3d.items():
            err_info = self.errors.get(name, {})
            quality = err_info.get('quality', 'unknown')
            data['points'][name] = {
                'x_mm': round(float(p3d[0]), 2),
                'y_mm': round(float(p3d[1]), 2),
                'z_mm': round(float(p3d[2]), 2),
                'observation_count': int(err_info.get('n_views', 0)),
                'mean_reprojection_error_px': round(float(err_info.get('mean', 0)), 2),
                'max_reprojection_error_px': round(float(err_info.get('max', 0)), 2),
                'triangulation_angle_deg':
                    round(float(err_info.get('tri_angle_deg', 0)), 1),
                'quality': quality,
            }

        # Statistics
        z_vals = [info['z_mm'] for info in data['points'].values()]
        if z_vals:
            data['stats'] = {
                'z_range_mm': f"{min(z_vals):.1f} ~ {max(z_vals):.1f}",
                'z_spread_mm': round(max(z_vals) - min(z_vals), 1),
                'z_std_mm': round(float(np.std(z_vals)), 1),
                'non_coplanar': max(z_vals) - min(z_vals) > 2.0,
                'good_points': sum(1 for p in data['points'].values()
                                   if p['quality'] == 'good'),
                'fair_points': sum(1 for p in data['points'].values()
                                   if p['quality'] == 'fair'),
            }

        # Rigid body check
        distances = self._rigid_body_check()
        if distances:
            data['pairwise_distances_mm'] = distances

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False,
                      allow_unicode=True, sort_keys=False)
        print(f"  -> {output_path}")

        # Summary table
        print(f"\n  {'Name':<14} {'X':>8} {'Y':>8} {'Z':>8} {'mean':>6} {'max':>6} "
              f"{'<deg':>5} {'n':>3} {'quality'}")
        print(f"  {'-'*70}")
        for name in sorted(data['points'].keys()):
            p = data['points'][name]
            print(f"  {name:<14} {p['x_mm']:>8.1f} {p['y_mm']:>8.1f} {p['z_mm']:>8.1f} "
                  f"{p['mean_reprojection_error_px']:>6.2f} {p['max_reprojection_error_px']:>6.2f} "
                  f"{p['triangulation_angle_deg']:>5.1f} {p['observation_count']:>3} "
                  f"{p['quality']}")

        if 'stats' in data:
            s = data['stats']
            print(f"\n  Z spread={s['z_spread_mm']} mm, "
                  f"good={s['good_points']}, fair={s['fair_points']}")

        return data


# ====================================================================
# 主函数
# ====================================================================

def main():
    import argparse
    p = argparse.ArgumentParser(
        description='Multi-view triangulation of aircraft marker 3D coordinates (v2)')
    p.add_argument('images', nargs='+', help='Image sequence with board + aircraft in frame')
    p.add_argument('--config', default='configs/experiment_config.yaml',
                   help='Camera calibration config file')
    p.add_argument('--output', '-o', default='configs/aircraft_points_measured.yaml',
                   help='Output YAML path')
    p.add_argument('--point-names', nargs='+',
                   help='Preset marker point names (e.g. nose_tip left_wing right_wing ...)')
    p.add_argument('--labels', help='Existing labels file for first frame (YAML, skip labeling step)')
    p.add_argument('--max-error', type=float, default=10.0,
                   help='Outlier view reprojection error threshold (px)')
    p.add_argument('--threshold', type=float, default=0.0,
                   help='Large circle detection threshold (0=auto-tune)')
    p.add_argument('--blob', action='store_true',
                   help='Use blob detector for board circles (recommended for 25mm)')
    args = p.parse_args()

    os.makedirs('output', exist_ok=True)

    # 读配置
    with open(args.config, encoding='utf-8') as f:
        exp = yaml.safe_load(f)
    cal = exp['calibration']
    K = np.array([[cal['fx'], 0, cal['cx']],
                   [0, cal['fy'], cal['cy']],
                   [0, 0, 1]], dtype=np.float64)
    dist = np.array(cal['dist'], dtype=np.float64)

    # 读图像
    print(f"Loading {len(args.images)} images...")
    images = []
    image_paths = []
    for pth in args.images:
        img = cv2.imread(pth)
        if img is None:
            print(f"  Warning: cannot read {pth}")
            continue
        images.append(img)
        image_paths.append(pth)
    print(f"  Successfully loaded {len(images)} images")

    if len(images) < 3:
        print("Need at least 3 images (5-7 recommended)"); sys.exit(1)

    # 初始化
    board_est = BoardPoseEstimator(K, dist, large_thresh=args.threshold,
                                    use_blob=True)  # CC detector
    gui = AircraftTriangulationGUI(images, image_paths, board_est, K, dist)

    # 步骤1: 标定板PnP
    gui.process_all_boards()
    if sum(gui.valids) < 3:
        print("Valid images < 3, triangulation cannot proceed")
        print("Please check: is the calibration board clearly visible in every photo?")
        sys.exit(1)

    # 步骤2: 注册标志点名称
    gui.register_point_names(preset_names=args.point_names)

    # 步骤3: 标注第一帧
    if not gui.label_first_frame(existing_labels=args.labels):
        print("Labeling cancelled"); sys.exit(0)

    if gui.point_count < 3:
        print("Need at least 3 marker points"); sys.exit(1)

    # 步骤4: 标注其余帧 (跳过KLT, 逐帧手动标注)
    gui.annotate_remaining_frames()

    # 步骤5: 三角测量
    gui.triangulate_all()

    # 步骤6: 异常视图剔除
    if gui.points3d:
        gui.remove_outlier_views(max_error_px=args.max_error)

    # 步骤6b: Huber BA 非线性精化
    if gui.points3d:
        gui._ba_refine_points()

    # 步骤7: 导出 (含刚体检查)
    if gui.points3d:
        gui.export_yaml(args.output)
        # 复制到标准位置
        import shutil
        shutil.copy(args.output, 'configs/aircraft_points.yaml')
        print("Updated configs/aircraft_points.yaml")
    else:
        print("\nTriangulation failed, no valid 3D points output.")
        print("Possible causes: insufficient observations, large board PnP error, point matching errors")


if __name__ == '__main__':
    main()
