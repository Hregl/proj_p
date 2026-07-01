"""
基于标志点的飞行器姿态估计 —— 多视图运动恢复结构（SfM）。

流程概览
--------
  Phase A —— 真实值采集（相机1，多视图）：
    1. 在所有图像中检测编码标志点（ArUco）。
    2. 通过本质矩阵 + 三角测量从最佳图像对初始化。
    3. 使用 PnP 增量注册剩余视图。
    4. 对新出现的标志点进行三角测量。
    5. 全局交会-投影交替法进行光束法平差。
    6. 将世界坐标系对齐到物理地面标志点。
    7. 计算三维标志点位置 → "真实值"。

  Phase B —— 单视图姿态推断（相机2）：
    使用 Phase A 的三维点通过 ``solvePnP`` 计算姿态。
    （参见 coded_marker.py 中的 ``CodedMarkerDetector.estimatePoseMulti``）

坐标系约定
----------
  - 所有旋转均使用 3×3 矩阵（R），而非 Rodrigues 向量。
  - 相机投影矩阵： P = K [R | t]，其中 K 为 3×3
    内参矩阵，[R | t] 将世界坐标系映射到相机坐标系。
  - 三维点以 **世界** 坐标系下的 (x, y, z) 存储。

参考文献
--------
  - Hartley & Zisserman, *Multiple View Geometry*, 第2版, 第9–10, 18章。
  - OpenCV ``cv2.solvePnPRansac``, ``cv2.triangulatePoints``,
    ``cv2.findEssentialMat``, ``cv2.recoverPose``。
"""

from __future__ import annotations

import itertools
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .coded_marker import CodedMarkerDetector
from .marker_detector import SLSMarkerDetector

# ---------------------------------------------------------------------------
# 类型别名
# ---------------------------------------------------------------------------

# 三维点索引（int）或映射 marker_id → (x, y, z) 的字典
Point3D = Tuple[float, float, float]

# ---------------------------------------------------------------------------
# View：SfM 重建中的单张图像
# ---------------------------------------------------------------------------


class View:
    """SfM 重建中的单张图像。"""

    __slots__ = (
        "name", "image", "markers", "centers",
        "R", "t", "registered",
    )

    def __init__(self, name: str, image: np.ndarray) -> None:
        self.name = name
        self.image = image
        # markers: {marker_id: corners_4x2}
        self.markers: Dict[int, np.ndarray] = {}
        # centers: {marker_id: (cx, cy)}
        self.centers: Dict[int, Tuple[float, float]] = {}
        # 相机在世界坐标系下的姿态：X_cam = R @ X_world + t
        self.R: Optional[np.ndarray] = None   # 3×3
        self.t: Optional[np.ndarray] = None   # 3×1
        self.registered: bool = False


# ===================================================================
# MultiViewSfM
# ===================================================================


class MultiViewSfM:
    """
    使用编码（ArUco）标志点的多视图运动恢复结构。

    Parameters
    ----------
    camera_matrix:
        3×3 相机内参矩阵 ``K``。
    dist_coeffs:
        镜头畸变系数（4, 5, 8 或 12 个元素）。
    marker_size_m:
        **单个 ArUco 标志点**的物理边长，单位为米。
        用于恢复绝对尺度。也可以通过
        ``set_scale_from_marker`` 后续设置。
    aruco_dict:
        ArUco 字典名称（与 ``CodedMarkerDetector`` 相同）。
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        marker_size_m: Optional[float] = None,
        aruco_dict: str = "4x4_50",
    ) -> None:
        self.K = camera_matrix.astype(np.float64)
        self.dist = np.asarray(dist_coeffs, dtype=np.float64).ravel()
        self.marker_size_m = marker_size_m
        self._scale = 1.0  # 在 BA 之后应用；真实尺度 = scale × 重建尺度

        # 检测器
        self._aruco = CodedMarkerDetector(
            dict_name=aruco_dict, refine_corners=True
        )
        self._circle = SLSMarkerDetector()

        # 数据
        self.views: List[View] = []
        self._points3d: Dict[int, Point3D] = {}  # marker_id → (x, y, z)
        self._point_observations: Dict[int, List[Tuple[int, int]]] = {}
        # ^ {marker_id: [(view_idx, corner_idx), ...]}

        # 计时
        self._timings: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # 步骤 1 —— 添加视图并检测标志点
    # ------------------------------------------------------------------

    def add_views(
        self,
        images: Sequence[np.ndarray],
        names: Optional[Sequence[str]] = None,
        detect_circles: bool = False,
    ) -> None:
        """
        向重建中添加图像并检测 ArUco 标志点。

        Args:
            images: BGR/灰度图像列表。
            names: 可选的图像名称（默认：``"view_0"``, …）。
            detect_circles: 是否同时检测非编码圆形点（较慢）。
        """
        if names is None:
            names = [f"view_{i}" for i in range(len(images))]

        for name, img in zip(names, images):
            view = View(name, img)
            # ArUco 检测
            aruco_markers, _ = self._aruco.detect(img)
            for m_id, corners, center, _side in aruco_markers:
                view.markers[m_id] = corners
                view.centers[m_id] = center

            # 可选的圆形点检测（用于密集跟踪）
            if detect_circles:
                circles, _ = self._circle.detectMarkers(img)
                # 圆形点使用负 ID 以避免冲突
                for i, ((cx, cy), _area) in enumerate(circles):
                    cid = -(i + 1)
                    view.centers[cid] = (cx, cy)
                    # 圆形点没有 corners 数组

            self.views.append(view)

        print(f"已添加 {len(images)} 个视图；"
              f"共 {sum(len(v.markers) for v in self.views)} 个 ArUco 检测结果")

    # ------------------------------------------------------------------
    # 步骤 2 —— 从最佳图像对初始化
    # ------------------------------------------------------------------

    def initialize(
        self,
        pair: Optional[Tuple[int, int]] = None,
        min_shared: int = 6,
        min_angle_deg: float = 3.0,
    ) -> bool:
        """
        从一对视图初始化重建。

        如果未指定 *pair*，则自动按
        (共享标志点数 × 三角测量角度) 最大化的原则选择最佳图像对。

        Returns:
            如果初始化成功则返回 ``True``。
        """
        t0 = time.perf_counter()

        if len(self.views) < 2:
            print("需要至少 2 个视图才能初始化。")
            return False

        # --- 自动选择最佳图像对 ------------------------------------
        if pair is None:
            pair = self._select_initial_pair(min_shared, min_angle_deg)
            if pair is None:
                print("找不到合适的初始图像对。"
                      "请检查标志点覆盖范围和基线。")
                return False

        v0, v1 = self.views[pair[0]], self.views[pair[1]]
        shared_ids = sorted(set(v0.centers.keys()) & set(v1.centers.keys()))
        # 仅保留 ArUco 标志点（正 ID）
        shared_ids = [sid for sid in shared_ids if sid >= 0]

        if len(shared_ids) < min_shared:
            print(f"仅有 {len(shared_ids)} 个共享标志点（需要 ≥{min_shared}）。")
            return False

        print(f"从 '{v0.name}' <-> '{v1.name}' 初始化 "
              f"({len(shared_ids)} 个共享标志点)")

        pts0 = np.array([v0.centers[sid] for sid in shared_ids], dtype=np.float64)
        pts1 = np.array([v1.centers[sid] for sid in shared_ids], dtype=np.float64)

        # 使用 RANSAC 计算本质矩阵
        E, inlier_mask = cv2.findEssentialMat(
            pts0, pts1, self.K, method=cv2.RANSAC,
            prob=0.999, threshold=1.0,
        )
        if E is None:
            print("估计本质矩阵失败。")
            return False

        inliers = inlier_mask.ravel().astype(bool)
        n_inliers = inliers.sum()
        print(f"  本质矩阵内点数: {n_inliers}/{len(shared_ids)}")

        if n_inliers < min_shared:
            print("本质矩阵 RANSAC 后内点太少。")
            return False

        # 恢复相对姿态（v1 在 v0 的坐标系中）
        pts0_in = pts0[inliers]
        pts1_in = pts1[inliers]
        n_pts, R_rel, t_rel, mask_recover = cv2.recoverPose(
            E, pts0_in, pts1_in, self.K
        )
        angle_deg = float(np.linalg.norm(cv2.Rodrigues(R_rel)[0]))
        print(f"  相对旋转: {np.rad2deg(angle_deg):.1f}°  "
              f"基线: {np.linalg.norm(t_rel):.3f}（未缩放）")

        # 将视图 0 设为世界原点
        v0.R = np.eye(3)
        v0.t = np.zeros((3, 1))
        v0.registered = True

        v1.R = R_rel
        v1.t = t_rel.reshape(3, 1)
        v1.registered = True

        # 对共享标志点进行三角测量
        P0 = self.K @ np.hstack([v0.R, v0.t])
        P1 = self.K @ np.hstack([v1.R, v1.t])

        pts0_f = pts0_in[mask_recover.ravel().astype(bool)].T.astype(np.float64)
        pts1_f = pts1_in[mask_recover.ravel().astype(bool)].T.astype(np.float64)
        shared_in = [sid for i, sid in enumerate(shared_ids)
                     if inliers[i] and mask_recover[i]]

        pts4d = cv2.triangulatePoints(P0, P1, pts0_f, pts1_f)
        pts3d = pts4d[:3] / pts4d[3]

        # 存储有效的三维点（在相机前方检查）
        valid_count = 0
        for i, sid in enumerate(shared_in):
            p = pts3d[:, i]
            # 检查点是否在相机 0 的前方
            if p[2] > 0:  # 在相机 0 前方
                # 变换到相机 1 坐标系
                p1 = R_rel @ p + t_rel.ravel()
                if p1[2] > 0:
                    self._points3d[sid] = (float(p[0]), float(p[1]), float(p[2]))
                    valid_count += 1

        print(f"  三角测量得到 {valid_count} 个三维点")

        self._timings["initialize"] = (time.perf_counter() - t0) * 1000
        return valid_count >= min_shared

    # ------------------------------------------------------------------
    # 步骤 3 —— 增量注册
    # ------------------------------------------------------------------

    def register_all(
        self,
        min_matches: int = 4,
        reproj_threshold: float = 3.0,
    ) -> int:
        """
        使用 PnP 结合现有三维点注册所有未注册视图。

        视图按"拥有已知三维位置的标志点数量"降序处理，
        以提高稳定性。每次成功注册后，对新观测到的标志点
        进行三角测量。

        Returns:
            新注册的视图数量。
        """
        t0 = time.perf_counter()
        newly_registered = 0

        while True:
            best_view = None
            best_matches = 0
            best_shared_ids: List[int] = []

            for view in self.views:
                if view.registered:
                    continue
                shared = [sid for sid in view.centers
                          if sid in self._points3d and sid >= 0]
                if len(shared) > best_matches:
                    best_matches = len(shared)
                    best_view = view
                    best_shared_ids = shared

            if best_view is None or best_matches < min_matches:
                break

            # --- PnP 注册 ---
            obj_pts = np.array(
                [self._points3d[sid] for sid in best_shared_ids],
                dtype=np.float64,
            )
            img_pts = np.array(
                [best_view.centers[sid] for sid in best_shared_ids],
                dtype=np.float64,
            )

            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_pts, img_pts, self.K, self.dist,
                flags=cv2.SOLVEPNP_EPNP,
                iterationsCount=200,
                reprojectionError=reproj_threshold,
                confidence=0.99,
            )

            if not success or (inliers is not None and len(inliers) < min_matches):
                print(f"  跳过 '{best_view.name}'：PnP 失败 "
                      f"({len(inliers) if inliers is not None else 0} 个内点)")
                # 标记为"已尝试但失败"—— 暂时将其设置为 registered=False
                # 且清空其姿态；当更多三维点加入后可能成功。
                best_view.R = None
                best_view.t = None
                best_view.registered = False
                # 跳过此视图，等更多点加入后重试
                print(f"    （将在添加更多点后重试）")
                break

            best_view.R, _ = cv2.Rodrigues(rvec)
            best_view.t = tvec.reshape(3, 1)
            best_view.registered = True
            newly_registered += 1

            n_inl = len(inliers) if inliers is not None else 0
            print(f"  已注册 '{best_view.name}' "
                  f"({n_inl}/{best_matches} 个内点)")

            # --- 对新观测到的标志点进行三角测量 ---
            self._triangulate_new_points(best_view)

        self._timings["register_all"] = (time.perf_counter() - t0) * 1000
        return newly_registered

    def _triangulate_new_points(self, new_view: View) -> int:
        """对 *new_view* 中且 ≥1 个其他已注册视图也观测到的标志点进行三角测量。"""
        registered_views = [v for v in self.views if v.registered]
        if len(registered_views) < 2:
            return 0

        new_count = 0
        for sid, center in new_view.centers.items():
            if sid < 0 or sid in self._points3d:
                continue  # 跳过圆形点和已知点

            # 找到也看到此标志点的其他已注册视图
            other_views = [
                v for v in registered_views
                if v is not new_view and sid in v.centers
            ]
            if not other_views:
                continue

            # 选择基线最大的其他视图
            best_v = max(
                other_views,
                key=lambda v: np.linalg.norm(new_view.t - v.t)
            )

            P1 = self.K @ np.hstack([new_view.R, new_view.t])
            P2 = self.K @ np.hstack([best_v.R, best_v.t])

            pt1 = np.array([[center]], dtype=np.float64)
            pt2 = np.array([[best_v.centers[sid]]], dtype=np.float64)

            pts4d = cv2.triangulatePoints(P1, P2, pt1.T, pt2.T)
            pts3d = pts4d[:3] / pts4d[3]

            p = pts3d[:, 0]
            # 检查是否在两台相机前方
            p1_cam = new_view.R @ p + new_view.t.ravel()
            p2_cam = best_v.R @ p + best_v.t.ravel()
            if p1_cam[2] > 0 and p2_cam[2] > 0:
                self._points3d[sid] = (float(p[0]), float(p[1]), float(p[2]))
                new_count += 1

        if new_count:
            print(f"    三角测量得到 {new_count} 个新标志点")
        return new_count

    # ------------------------------------------------------------------
    # 步骤 4 —— 光束法平差（交会-投影交替法 + 稀疏 LM）
    # ------------------------------------------------------------------

    def bundle_adjust(
        self,
        iterations: int = 10,
        use_sparse_lm: bool = True,
        verbose: bool = True,
    ) -> float:
        """
        全局光束法平差，优化相机姿态和三维点。

        分两阶段：
        1. **交会-投影交替法**（快速，始终执行）：
           交替执行每台相机的 PnP 求解和
           每个三维点的重三角测量。
        2. **稀疏 Levenberg-Marquardt**（可选，更精细）：
           使用 ``scipy.optimize.least_squares`` 联合优化
           所有参数以最小化重投影误差。

        Returns:
            最终的平均重投影误差（像素）。
        """
        t0 = time.perf_counter()

        reg_views = [v for v in self.views if v.registered]
        if len(reg_views) < 2 or len(self._points3d) < 4:
            print("视图或点数太少，无法进行 BA。")
            return float("inf")

        # --- 阶段 1：交会-投影交替法 -----------------------------
        for it in range(iterations):
            # 投影（Resection）：通过 PnP 精化每台相机的姿态
            for view in reg_views:
                shared = [sid for sid in view.centers
                          if sid in self._points3d and sid >= 0]
                if len(shared) < 4:
                    continue
                obj_pts = np.array(
                    [self._points3d[sid] for sid in shared], dtype=np.float64
                )
                img_pts = np.array(
                    [view.centers[sid] for sid in shared], dtype=np.float64
                )
                # 在此阶段使用迭代精化（非 RANSAC），因为我们相信
                # 初始姿态和三维点是可靠的
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts, self.K, self.dist,
                    rvec=cv2.Rodrigues(view.R)[0],
                    tvec=view.t,
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if ok:
                    view.R, _ = cv2.Rodrigues(rvec)
                    view.t = tvec.reshape(3, 1)

            # 交会（Intersection）：从所有观测视图精化每个三维点
            for sid in list(self._points3d.keys()):
                observing_views = [
                    v for v in reg_views if sid in v.centers
                ]
                if len(observing_views) < 2:
                    continue
                # 从所有视图同时进行三角测量（DLT 法）
                self._points3d[sid] = self._triangulate_from_views(
                    sid, observing_views
                )

        # --- 阶段 2：稀疏 LM（可选）-------------------------------
        if use_sparse_lm and len(self._points3d) >= 4:
            try:
                final_err = self._sparse_bundle_adjust(verbose)
            except Exception as exc:
                if verbose:
                    print(f"  稀疏 BA 失败 ({exc})；"
                          f"使用交会-投影交替法结果。")
                final_err = self._compute_mean_reproj_error()
        else:
            final_err = self._compute_mean_reproj_error()

        self._timings["bundle_adjust"] = (time.perf_counter() - t0) * 1000
        if verbose:
            n_views = len(reg_views)
            n_pts = len(self._points3d)
            t_ms = self._timings["bundle_adjust"]
            print(f"BA：{n_views} 个视图, {n_pts} 个点 → "
                  f"重投影误差 = {final_err:.3f} px  ({t_ms:.0f} ms)")

        return final_err

    def _triangulate_from_views(
        self, marker_id: int, views: List[View]
    ) -> Point3D:
        """从所有观测到该点的视图进行三角测量（DLT 法）。"""
        n = len(views)
        if n < 2:
            return self._points3d.get(marker_id, (0.0, 0.0, 0.0))

        # 构建线性系统 A X = 0，其中 X 为三维点
        A = np.zeros((2 * n, 4), dtype=np.float64)
        for i, view in enumerate(views):
            cx, cy = view.centers[marker_id]
            P = self.K @ np.hstack([view.R, view.t])
            A[2 * i]     = cx * P[2] - P[0]
            A[2 * i + 1] = cy * P[2] - P[1]

        _, _, Vt = np.linalg.svd(A, full_matrices=False)
        X = Vt[-1]
        p = X[:3] / X[3]

        # 检查是否在所有相机前方
        all_front = True
        for view in views:
            pc = view.R @ p + view.t.ravel()
            if pc[2] <= 0:
                all_front = False
                break

        if not all_front:
            # 回退到原始点
            return self._points3d.get(marker_id, (float(p[0]), float(p[1]), float(p[2])))

        return (float(p[0]), float(p[1]), float(p[2]))

    # ------------------------------------------------------------------
    # 稀疏 Levenberg-Marquardt 光束法平差
    # ------------------------------------------------------------------

    def _sparse_bundle_adjust(self, verbose: bool = True) -> float:
        """
        使用稀疏雅可比矩阵最小化重投影误差。

        仅优化相机外参和三维点位置；
        内参保持不变。利用每个观测仅依赖于一台相机和
        一个三维点这一事实——雅可比矩阵具有块对角结构，
        从而利用稀疏性。
        """
        reg_views = [v for v in self.views if v.registered]
        point_ids = sorted(self._points3d.keys())

        n_cameras = len(reg_views)
        n_points = len(point_ids)
        n_params = 6 * n_cameras + 3 * n_points  # 每相机 rvec(3) + t(3)，每点 (x,y,z)

        # 构建观测列表
        # 每个观测：(camera_idx, point_idx, u, v)
        observations: List[Tuple[int, int, float, float]] = []
        pt_to_idx = {pid: i for i, pid in enumerate(point_ids)}

        for ci, view in enumerate(reg_views):
            for pid, (u, v) in view.centers.items():
                if pid in pt_to_idx:
                    observations.append((ci, pt_to_idx[pid], u, v))
        n_obs = len(observations)

        if n_obs < n_params:
            if verbose:
                print(f"  跳过稀疏 BA：{n_obs} 个观测 < {n_params} 个参数")
            return self._compute_mean_reproj_error()

        if verbose:
            print(f"  稀疏 BA：{n_cameras} 台相机, {n_points} 个点, "
                  f"{n_obs} 个观测 → {n_params} 个参数")

        # 初始参数向量
        x0 = np.zeros(n_params, dtype=np.float64)

        for ci, view in enumerate(reg_views):
            rvec = cv2.Rodrigues(view.R)[0].ravel()
            x0[6 * ci: 6 * ci + 3] = rvec
            x0[6 * ci + 3: 6 * ci + 6] = view.t.ravel()

        for pi, pid in enumerate(point_ids):
            offset = 6 * n_cameras + 3 * pi
            x0[offset: offset + 3] = self._points3d[pid]

        # 构建稀疏性结构
        jac_sparsity = self._build_sparsity(
            n_cameras, n_points, observations, n_params
        )

        # 运行 LM
        result = least_squares(
            self._reproj_residuals,
            x0,
            jac_sparsity=jac_sparsity,
            method="trf",           # Trust Region Reflective（支持稀疏）
            loss="soft_l1",         # 对离群点鲁棒
            f_scale=2.0,            # ~2 px 的内点阈值
            max_nfev=50,
            verbose=1 if verbose else 0,
            args=(reg_views, point_ids, observations),
        )

        # 解包结果
        x_opt = result.x
        for ci, view in enumerate(reg_views):
            rvec = x_opt[6 * ci: 6 * ci + 3]
            view.R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
            view.t = x_opt[6 * ci + 3: 6 * ci + 6].reshape(3, 1)

        for pi, pid in enumerate(point_ids):
            offset = 6 * n_cameras + 3 * pi
            self._points3d[pid] = (
                float(x_opt[offset]),
                float(x_opt[offset + 1]),
                float(x_opt[offset + 2]),
            )

        return float(np.sqrt(result.cost * 2 / n_obs))

    def _reproj_residuals(
        self,
        params: np.ndarray,
        reg_views: List[View],
        point_ids: List[int],
        observations: List[Tuple[int, int, float, float]],
    ) -> np.ndarray:
        """稀疏 BA 的残差函数。"""
        n_cameras = len(reg_views)
        residuals = np.zeros(2 * len(observations), dtype=np.float64)

        # 解包相机参数
        cam_R = []
        cam_t = []
        for ci in range(n_cameras):
            rvec = params[6 * ci: 6 * ci + 3].reshape(3, 1)
            R, _ = cv2.Rodrigues(rvec)
            cam_R.append(R)
            cam_t.append(params[6 * ci + 3: 6 * ci + 6].reshape(3, 1))

        for oi, (ci, pi, u, v) in enumerate(observations):
            offset = 6 * n_cameras + 3 * pi
            pt3d = params[offset: offset + 3]

            # 投影
            pc = cam_R[ci] @ pt3d + cam_t[ci].ravel()
            if pc[2] <= 1e-9:
                residuals[2 * oi] = 1e6
                residuals[2 * oi + 1] = 1e6
                continue

            up = pc[0] / pc[2] * self.K[0, 0] + self.K[0, 2]
            vp = pc[1] / pc[2] * self.K[1, 1] + self.K[1, 2]

            residuals[2 * oi] = up - u
            residuals[2 * oi + 1] = vp - v

        return residuals

    @staticmethod
    def _build_sparsity(
        n_cameras: int,
        n_points: int,
        observations: List[Tuple[int, int, float, float]],
        n_params: int,
    ) -> lil_matrix:
        """构建 BA 的稀疏雅可比结构。"""
        n_obs = len(observations)
        J = lil_matrix((2 * n_obs, n_params))

        for oi, (ci, pi, _u, _v) in enumerate(observations):
            # 相机列：每台相机 6 个参数
            J[2 * oi,     6 * ci: 6 * ci + 6] = 1
            J[2 * oi + 1, 6 * ci: 6 * ci + 6] = 1
            # 点列：每个点 3 个参数
            offset = 6 * n_cameras + 3 * pi
            J[2 * oi,     offset: offset + 3] = 1
            J[2 * oi + 1, offset: offset + 3] = 1

        return J

    # ------------------------------------------------------------------
    # 尺度解析
    # ------------------------------------------------------------------

    def set_scale_from_marker(
        self, marker_id: int, physical_size_m: float
    ) -> float:
        """
        从已知尺寸的标志点计算绝对尺度。

        将 *marker_id* 的相邻角点在重建三维空间中的距离
        与 *physical_size_m* 进行比较。

        Returns:
            计算得到的尺度因子（调用后 ≈ 1.0）。
        """
        if marker_id not in self._points3d:
            raise ValueError(f"标志点 {marker_id} 不在三维点集中。")

        # 我们没有存储角点的三维坐标；使用已注册视图中
        # 二维检测结果的边长来推算。
        side_lengths = []
        for view in self.views:
            if not view.registered or marker_id not in view.markers:
                continue
            corners = view.markers[marker_id]
            # 使用标志点中心和 PnP 姿态反投影已知尺寸的角点。
            rvec = cv2.Rodrigues(view.R)[0]
            tvec = view.t
            # 标志点位于平面上；对此标志点使用 solvePnP 结果
            half = physical_size_m / 2.0
            obj_pts = np.array([
                [-half,  half, 0],
                [ half,  half, 0],
                [ half, -half, 0],
                [-half, -half, 0],
            ], dtype=np.float32)
            img_pts, _ = cv2.projectPoints(
                obj_pts, rvec, tvec, self.K, self.dist
            )
            # 与检测到的角点比较
            detected = corners.astype(np.float32).reshape(4, 2)
            proj = img_pts.reshape(4, 2)
            # 将期望边长与实际边长的比值作为尺度
            for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                expected = np.linalg.norm(proj[a] - proj[b])
                actual = np.linalg.norm(detected[a] - detected[b])
                if actual > 0 and expected > 0:
                    side_lengths.append(expected / actual)

        if not side_lengths:
            raise RuntimeError(
                f"标志点 {marker_id} 在任何已注册视图中均不可见。"
            )

        # 尺度因子用于调整三维坐标
        scale = float(np.median(side_lengths))
        # 不自动应用；由调用者决定
        return scale

    def apply_scale(self, scale: float) -> None:
        """将所有三维坐标和相机平移乘以 *scale*。"""
        self._scale *= scale
        for sid in self._points3d:
            x, y, z = self._points3d[sid]
            self._points3d[sid] = (x * scale, y * scale, z * scale)
        for view in self.views:
            if view.registered and view.t is not None:
                view.t *= scale

    # ------------------------------------------------------------------
    # 坐标系对齐
    # ------------------------------------------------------------------

    def align_to_ground(
        self,
        ground_marker_ids: List[int],
        ground_plane_points: Optional[List[Tuple[float, float, float]]] = None,
    ) -> np.ndarray:
        """
        对齐世界坐标系，使地面标志点定义 XY 平面（Z=0），
        且其质心位于原点。

        如果提供了 *ground_plane_points*（以米为单位的物理测量值），
        则应用相似变换将重建点变换到该度量坐标系中。

        Returns:
            应用的 4×4 齐次变换矩阵。
        """
        if len(ground_marker_ids) < 3:
            raise ValueError("需要 ≥3 个地面标志点来定义一个平面。")

        # 收集地面标志点的重建三维位置
        pts_rec = np.array(
            [self._points3d[sid] for sid in ground_marker_ids
             if sid in self._points3d],
            dtype=np.float64,
        )
        if len(pts_rec) < 3:
            raise RuntimeError("少于 3 个地面标志点拥有三维坐标。")

        # --- 对地面标志点拟合平面（在重建坐标系中）---------
        centroid = pts_rec.mean(axis=0)
        pts_centered = pts_rec - centroid
        _, _, Vt = np.linalg.svd(pts_centered, full_matrices=False)
        normal = Vt[-1]   # 平面法向量（最小奇异值向量）
        if normal[2] < 0:
            normal = -normal  # 指向上方
        normal = normal / np.linalg.norm(normal)

        # --- 构建将法向量 → (0, 0, 1) 的旋转 -----------------
        z_axis = np.array([0.0, 0.0, 1.0])
        axis = np.cross(normal, z_axis)
        angle = math.acos(np.clip(np.dot(normal, z_axis), -1.0, 1.0))
        if np.linalg.norm(axis) < 1e-12:
            R_align = np.eye(3)
        else:
            axis = axis / np.linalg.norm(axis)
            rvec = axis * angle
            R_align, _ = cv2.Rodrigues(rvec)

        t_align = -R_align @ centroid

        # 构建齐次变换：T_world_new = [R_align | t_align]
        T = np.eye(4)
        T[:3, :3] = R_align
        T[:3, 3] = t_align

        # 应用到所有三维点
        for sid in list(self._points3d.keys()):
            p = np.array(self._points3d[sid])
            p_new = R_align @ p + t_align
            self._points3d[sid] = (float(p_new[0]), float(p_new[1]), float(p_new[2]))

        # 应用到相机姿态
        for view in self.views:
            if not view.registered:
                continue
            # 相机姿态：X_cam = R @ X_world + t
            # 变换后：X_world_new = R_align @ X_world_old + t_align
            # 因此：R_new @ X_world_new + t_new = R_old @ X_world_old + t_old
            R_old, t_old = view.R, view.t.ravel()
            view.R = R_old @ R_align.T
            view.t = (t_old - view.R @ t_align).reshape(3, 1)

        # --- 可选：将相似变换应用到度量坐标系 ---------------
        if ground_plane_points is not None:
            T_sim = self._similarity_align(ground_marker_ids, ground_plane_points)
            T = T_sim @ T

        print(f"已应用地面对齐。  "
              f"世界坐标系中的平面法向量：({normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f})")

        return T

    def _similarity_align(
        self,
        marker_ids: List[int],
        target_points: List[Tuple[float, float, float]],
    ) -> np.ndarray:
        """计算对齐到目标坐标的相似变换（7 自由度）。"""
        src = np.array(
            [self._points3d[mid] for mid in marker_ids
             if mid in self._points3d],
            dtype=np.float64,
        )
        tgt = np.array(target_points, dtype=np.float64)

        if len(src) != len(tgt) or len(src) < 3:
            print("警告：用于相似对齐的点数不足。")
            return np.eye(4)

        # 估计相似变换（尺度、旋转、平移）
        # 不包含反射的 Procrustes 分析
        src_centroid = src.mean(axis=0)
        tgt_centroid = tgt.mean(axis=0)
        src_c = src - src_centroid
        tgt_c = tgt - tgt_centroid

        # 尺度
        scale = np.sqrt(
            np.sum(tgt_c ** 2) / np.sum(src_c ** 2)
        )

        # 通过 SVD 求旋转
        H = src_c.T @ tgt_c
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T

        t = tgt_centroid - scale * R @ src_centroid

        # 应用
        for sid in list(self._points3d.keys()):
            p = np.array(self._points3d[sid])
            p_new = scale * R @ p + t
            self._points3d[sid] = (float(p_new[0]), float(p_new[1]), float(p_new[2]))

        for view in self.views:
            if not view.registered:
                continue
            R_old, t_old = view.R, view.t.ravel()
            # X_new = scale * R @ X_old + t
            # 推导：C_new = scale * R @ C_old + t,
            #   t_new = -R_new @ C_new, R_new = R_old @ R^T
            view.R = R_old @ R.T
            view.t = (scale * t_old - view.R @ t).reshape(3, 1)

        T = np.eye(4)
        T[:3, :3] = scale * R
        T[:3, 3] = t

        print(f"  相似对齐：scale={scale:.4f}")
        return T

    # ------------------------------------------------------------------
    # 飞行器姿态提取
    # ------------------------------------------------------------------

    def get_rigid_transform(
        self,
        marker_ids: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算一组标志点在其重建三维位置与用户定义的
        局部坐标之间的刚体变换 (R, t)。

        用于确定**飞行器姿态**：将飞行器标志点的已知局部坐标
        （在飞行器机体坐标系中）与其重建的世界坐标对齐，
        得到飞行器在世界坐标系中的位置和朝向。

        Args:
            marker_ids: 飞行器上 ArUco 标志点的 ID 列表。

        Returns:
            ``(R, t)``，满足 ``X_world = R @ X_local + t``。
            如果不到 3 个标志点拥有三维坐标，两者均为 ``None``。
        """
        pts_world = []
        valid_ids = []
        for mid in marker_ids:
            if mid in self._points3d:
                pts_world.append(self._points3d[mid])
                valid_ids.append(mid)

        if len(valid_ids) < 3:
            print(f"仅有 {len(valid_ids)} 个飞行器标志点拥有三维坐标 "
                  f"（需要 ≥3 个才能进行刚体变换）。")
            return None, None

        pts_world_arr = np.array(pts_world, dtype=np.float64)

        # 局部坐标：使用质心 + PCA 定义局部坐标系。
        # 在实际应用中，用户会提供测量得到的局部坐标。
        # 这里提供默认值：质心为原点，主方向为基向量。
        centroid = pts_world_arr.mean(axis=0)
        pts_centered = pts_world_arr - centroid

        # 如果仅有世界位置（没有独立的局部模型），
        # R 为单位矩阵，t = 质心。这给出飞行器在世界坐标系中的位置。
        R = np.eye(3)
        t = centroid.reshape(3,)

        print(f"由 {len(valid_ids)} 个标志点得到的飞行器姿态："
              f"t = ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}) m")
        return R, t

    def get_aircraft_pose_pnp(
        self,
        aircraft_marker_ids: List[int],
        aircraft_local_coords: List[Tuple[float, float, float]],
        image: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Phase B：通过 PnP 进行单视图飞行器姿态估计。

        给定飞行器标志点的重建三维世界坐标（来自 Phase A），
        以及来自相机 2 的单张图像，计算飞行器在地面坐标系下的姿态。

        Args:
            aircraft_marker_ids: 飞行器上的 ArUco ID 列表。
            aircraft_local_coords: 对应的 (x, y, z) 坐标，
                在飞行器机体坐标系中（米，物理测量值）。
            image: 来自相机 2 的单张图像。

        Returns:
            ``(R_aircraft_in_world, t_aircraft_in_world)``，
            失败时返回 ``(None, None)``。
        """
        # 检测标志点
        aruco_markers, _ = self._aruco.detect(image)
        img_pts = []
        obj_pts = []
        for m_id, corners, _, _ in aruco_markers:
            if m_id in aircraft_marker_ids:
                idx = aircraft_marker_ids.index(m_id)
                lx, ly, lz = aircraft_local_coords[idx]
                # 使用全部 4 个角点
                for k in range(4):
                    img_pts.append(corners[k])
                    obj_pts.append([lx, ly, lz])

        if len(obj_pts) < 4:
            print("图像中检测到的飞行器标志点角点数量不足。")
            return None, None

        img_pts_arr = np.array(img_pts, dtype=np.float64)
        obj_pts_arr = np.array(obj_pts, dtype=np.float64)

        # 首先获取相机在世界坐标系中的姿态（使用地面标志点），
        # 然后将飞行器点变换到世界坐标系。
        # 为了简化，这里直接使用世界坐标三维点（来自 Phase A）
        # 进行 PnP 求解：

        # 为检测到的飞行器标志点获取世界三维坐标
        obj_pts_world = []
        img_pts_world = []
        for m_id, corners, _, _ in aruco_markers:
            if m_id in self._points3d:
                for k in range(4):
                    img_pts_world.append(corners[k])
                    obj_pts_world.append(self._points3d[m_id])

        if len(obj_pts_world) < 12:  # 至少 3 个标志点 × 4 个角点
            print("拥有已知三维坐标的飞行器标志点数量不足。")
            return None, None

        obj_pts_w = np.array(obj_pts_world, dtype=np.float64)
        img_pts_w = np.array(img_pts_world, dtype=np.float64)

        success, rvec, tvec, _ = cv2.solvePnPRansac(
            obj_pts_w, img_pts_w, self.K, self.dist,
            flags=cv2.SOLVEPNP_EPNP,
            iterationsCount=100,
            reprojectionError=2.0,
        )

        if not success:
            return None, None

        R_cam, _ = cv2.Rodrigues(rvec)
        # 相机姿态：X_cam = R_cam @ X_world + tvec
        # 我们需要飞行器 → 世界的变换。飞行器标志点在飞行器
        # 机体坐标系中有已知位置。其世界坐标在 self._points3d 中。
        # 我们已经对齐到地面坐标系。
        # 返回飞行器机体坐标系的变换矩阵。

        # 使用 Procrustes 分析在 aircraft_local_coords 和
        # 其对应的世界三维点之间求解：
        matched_local = []
        matched_world = []
        for i, mid in enumerate(aircraft_marker_ids):
            if mid in self._points3d:
                matched_local.append(aircraft_local_coords[i])
                matched_world.append(self._points3d[mid])

        if len(matched_local) < 3:
            return None, None

        loc = np.array(matched_local, dtype=np.float64)
        wrl = np.array(matched_world, dtype=np.float64)

        loc_c = loc - loc.mean(axis=0)
        wrl_c = wrl - wrl.mean(axis=0)
        H = loc_c.T @ wrl_c
        U, _, Vt = np.linalg.svd(H)
        R_ac = Vt.T @ U.T
        if np.linalg.det(R_ac) < 0:
            Vt[-1] *= -1
            R_ac = Vt.T @ U.T

        t_ac = wrl.mean(axis=0) - R_ac @ loc.mean(axis=0)

        return R_ac, t_ac.reshape(3,)

    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------

    def _select_initial_pair(
        self, min_shared: int, min_angle_deg: float
    ) -> Optional[Tuple[int, int]]:
        """对所有视图对评分并返回最佳的一对。"""
        best_score = -1.0
        best_pair = None
        n = len(self.views)

        for i, j in itertools.combinations(range(n), 2):
            va, vb = self.views[i], self.views[j]
            shared = [sid for sid in va.centers
                      if sid in vb.centers and sid >= 0]
            if len(shared) < min_shared:
                continue

            pts_a = np.array([va.centers[sid] for sid in shared], dtype=np.float64)
            pts_b = np.array([vb.centers[sid] for sid in shared], dtype=np.float64)

            try:
                E, mask = cv2.findEssentialMat(
                    pts_a, pts_b, self.K, method=cv2.RANSAC,
                    prob=0.999, threshold=1.0,
                )
                if E is None:
                    continue
                inliers = mask.ravel().astype(bool).sum()
                if inliers < min_shared:
                    continue
                _, R, _, _ = cv2.recoverPose(E, pts_a, pts_b, self.K)
                angle = np.linalg.norm(cv2.Rodrigues(R)[0])
                if angle < np.deg2rad(min_angle_deg):
                    continue
                score = inliers * angle
                if score > best_score:
                    best_score = score
                    best_pair = (i, j)
            except cv2.error:
                continue

        return best_pair

    def _compute_mean_reproj_error(self) -> float:
        """计算所有已注册视图的平均重投影误差。"""
        total_err = 0.0
        n_obs = 0
        for view in self.views:
            if not view.registered:
                continue
            for sid, (u, v) in view.centers.items():
                if sid not in self._points3d or sid < 0:
                    continue
                p3d = np.array(self._points3d[sid], dtype=np.float64)
                pc = view.R @ p3d + view.t.ravel()
                if pc[2] <= 0:
                    continue
                up = pc[0] / pc[2] * self.K[0, 0] + self.K[0, 2]
                vp = pc[1] / pc[2] * self.K[1, 1] + self.K[1, 2]
                total_err += math.hypot(up - u, vp - v)
                n_obs += 1
        return total_err / n_obs if n_obs > 0 else float("inf")

    @property
    def points_3d(self) -> Dict[int, Point3D]:
        """重建的三维标志点位置（世界坐标系）。"""
        return dict(self._points3d)

    @property
    def camera_poses(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """所有已注册视图的 ``[(R, t), …]``。"""
        return [
            (v.R, v.t) for v in self.views if v.registered
        ]

    def summary(self) -> str:
        """打印人类可读的重建摘要。"""
        n_reg = sum(1 for v in self.views if v.registered)
        n_total = len(self.views)
        n_pts = len(self._points3d)
        err = self._compute_mean_reproj_error()
        lines = [
            f"SfM 重建摘要",
            f"  视图:     {n_reg}/{n_total} 已注册",
            f"  三维点:   {n_pts}",
            f"  重投影误差: {err:.3f} px",
        ]
        for k, v in self._timings.items():
            lines.append(f"  {k}: {v:.0f} ms")
        return "\n".join(lines)


# ===================================================================
# 合成数据测试（注入 GT 投影 → 测试 SfM 数学）
# ===================================================================
def _generate_synthetic_scene(
    n_markers: int = 15,
    n_views: int = 6,
    noise_px: float = 0.5,
    image_size: Tuple[int, int] = (1280, 720),
) -> Tuple[
    List[np.ndarray],          # 图像（已绘制标志点位置）
    np.ndarray,                # K
    np.ndarray,                # dist_coeffs
    Dict[int, Point3D],        # GT 三维点
    List[Tuple[np.ndarray, np.ndarray]],  # GT 相机姿态 (R, t)
]:
    """
    为 SfM 测试生成合成数据。

    不通过渲染和重新检测 ArUco 标志点的方式（这种方式不稳定），
    而是将已知三维点通过已知相机投影，添加噪声后直接填充 View 对象。

    Returns:
        (images, K, dist, gt_points, gt_poses)
        图像仅用于调试可视化。
    """
    rng = np.random.default_rng(42)

    w, h = image_size
    K = np.array([[1000, 0, w / 2], [0, 1000, h / 2], [0, 0, 1]],
                 dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)

    # 真实值三维点 ----------------------------------------------
    gt_points: Dict[int, Point3D] = {}
    for i in range(n_markers):
        if i < n_markers // 2:
            x = (rng.random() - 0.5) * 0.6
            y = (rng.random() - 0.5) * 0.6
            z = rng.random() * 0.02
        else:
            x = (rng.random() - 0.5) * 0.3
            y = (rng.random() - 0.5) * 0.3
            z = 0.15 + rng.random() * 0.1
        gt_points[i] = (float(x), float(y), float(z))

    # 真实值相机姿态 -------------------------------------------
    gt_poses: List[Tuple[np.ndarray, np.ndarray]] = []
    rvecs = []
    tvecs = []
    for vi in range(n_views):
        angle = 2 * math.pi * vi / n_views
        radius = 0.7 + rng.random() * 0.2
        cam_x = radius * math.cos(angle)
        cam_y = radius * math.sin(angle * 0.9)
        cam_z = 0.25 + 0.15 * math.sin(angle * 0.7)

        cam_pos = np.array([cam_x, cam_y, cam_z])
        look_at = np.array([0.0, 0.0, 0.08])
        z_axis = look_at - cam_pos
        z_axis = z_axis / np.linalg.norm(z_axis)
        x_axis = np.cross(np.array([0.0, 1.0, 0.0]), z_axis)
        if np.linalg.norm(x_axis) < 1e-6:
            x_axis = np.cross(np.array([1.0, 0.0, 0.0]), z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        R = np.vstack([x_axis, y_axis, z_axis])  # 行 = 世界坐标系中的相机轴
        t = (-R @ cam_pos).reshape(3, 1)

        gt_poses.append((R, t))
        rvecs.append(cv2.Rodrigues(R)[0].ravel())
        tvecs.append(t.ravel())

    # 生成标注后的图像（仅用于可视化）--------------------------
    images = []
    for vi in range(n_views):
        img = np.full((h, w, 3), 240, dtype=np.uint8)
        R, t = gt_poses[vi]
        rvec = rvecs[vi]
        tvec = tvecs[vi]

        for mid, (mx, my, mz) in gt_points.items():
            # 使用 cv2.projectPoints 投影（正确处理畸变）
            pt3d = np.array([[[mx, my, mz]]], dtype=np.float32)
            pt2d, _ = cv2.projectPoints(pt3d, rvec.reshape(3, 1),
                                        tvec.reshape(3, 1), K.astype(np.float32), None)
            u, v = pt2d[0, 0]

            if not (-50 <= u < w + 50 and -50 <= v < h + 50):
                continue

            # 颜色：红色=地面，蓝色=飞行器
            color = (0, 0, 220) if mid < n_markers // 2 else (220, 0, 0)
            cv2.circle(img, (int(u), int(v)), 5, color, -1)
            cv2.putText(img, str(mid), (int(u) + 8, int(v) - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # 相机标签
        cv2.putText(img, f"View {vi}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        images.append(img)

    return images, K, dist, gt_points, gt_poses


def _inject_projections(
    sfm: MultiViewSfM,
    gt_points: Dict[int, Point3D],
    gt_poses: List[Tuple[np.ndarray, np.ndarray]],
    noise_px: float = 0.5,
) -> None:
    """
    用真实值投影（+ 可选高斯噪声）填充 SfM View 对象。

    此方法跳过 ArUco 检测，独立测试核心 SfM 流水线
    （初始化、注册、BA）。
    """
    rng = np.random.default_rng(123)
    n_markers = len(gt_points)

    for vi, (R, t) in enumerate(gt_poses):
        if vi >= len(sfm.views):
            sfm.views.append(View(f"view_{vi}",
                np.zeros((720, 1280, 3), dtype=np.uint8)))
        view = sfm.views[vi]  # ← 操作实际存储的 View 对象
        view.name = f"view_{vi}"
        view.markers.clear()
        view.centers.clear()

        rvec = cv2.Rodrigues(R)[0]
        tvec = t.reshape(3, 1)

        for mid, (mx, my, mz) in gt_points.items():
            pt3d = np.array([[[mx, my, mz]]], dtype=np.float32)
            pt2d, _ = cv2.projectPoints(
                pt3d, rvec, tvec,
                sfm.K.astype(np.float32),
                sfm.dist.astype(np.float32),
            )
            u_raw, v_raw = float(pt2d[0, 0, 0]), float(pt2d[0, 0, 1])

            # 向中心坐标添加高斯噪声
            u = u_raw + float(rng.normal(0, noise_px))
            v = v_raw + float(rng.normal(0, noise_px))

            h_img, w_img = 720, 1280
            if view.image is not None:
                h_img, w_img = view.image.shape[:2]

            if 0 <= u < w_img and 0 <= v < h_img:
                view.centers[mid] = (u, v)
                # 伪造 4 个角点（中心周围约 30 px 的正方形）
                s = 15.0
                corners = np.array([
                    [u - s, v - s], [u + s, v - s],
                    [u + s, v + s], [u - s, v + s],
                ], dtype=np.float32)
                view.markers[mid] = corners


# ===================================================================
# 主函数 — 合成数据测试
# ===================================================================

# ===================================================================
# （测试代码已提取到 tests/test_sfm_pipeline.py）
# ===================================================================
