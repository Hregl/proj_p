"""
基于自然特征的运动恢复结构（无需编码标记）。

支持顺序图像序列的增量式 SfM：
  1. SIFT/ORB 特征提取
  2. 顺序帧间匹配 + 比率检验 + 几何验证
  3. 本质矩阵初始化 → 增量 PnP 注册
  4. 三角测量 + 捆绑调整
  5. RANSAC 地面平面拟合
  6. 飞机点云分割（按高度 + 空间聚类）

与 marker-based pipeline 不同，本模块使用自然图像特征
（SIFT 关键点 + 描述子），不需要 ArUco 等编码标记。
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

# ---------------------------------------------------------------------------
# 特征视图
# ---------------------------------------------------------------------------

class FeatureView:
    """存储单张图像的特征和（可选的）已注册相机位姿。"""

    __slots__ = ("name", "image", "keypoints", "descriptors",
                 "R", "t", "registered", "point_ids")

    def __init__(self, name: str, image: np.ndarray):
        self.name = name
        self.image = image
        self.keypoints: List[cv2.KeyPoint] = []
        self.descriptors: Optional[np.ndarray] = None  # (N, 128) for SIFT
        # 相机位姿（世界系中）
        self.R: Optional[np.ndarray] = None
        self.t: Optional[np.ndarray] = None
        self.registered: bool = False
        # 每个关键点对应的三维点 ID（-1 = 尚无对应）
        self.point_ids: List[int] = []


# ---------------------------------------------------------------------------
# 特征匹配对
# ---------------------------------------------------------------------------

class MatchPair:
    """一对视图之间的特征匹配关系。"""

    def __init__(self, idx_a: int, idx_b: int):
        self.idx_a = idx_a
        self.idx_b = idx_b
        self.matches: List[cv2.DMatch] = []      # 原始匹配
        self.inlier_matches: List[cv2.DMatch] = []  # 几何验证后的内点
        self.E: Optional[np.ndarray] = None       # 本质矩阵
        self.F: Optional[np.ndarray] = None       # 基础矩阵


# ===================================================================
# FeatureBasedSfM
# ===================================================================


class FeatureBasedSfM:
    """
    基于自然特征的增量式运动恢复结构。

    工作流程（顺序模式）：
      1. extract_features()          对所有图像提取 SIFT/ORB
      2. match_sequential()          图像 i <-> i+1 顺序匹配
      3. initialize(pair)            从第一对初始化
      4. register_all_sequential()   逐个注册剩余视图
      5. bundle_adjust()             全局捆绑调整
      6. fit_ground_plane()          拟合地面平面
      7. segment_aircraft()          分割飞机点云

    Parameters
    ----------
    K : np.ndarray
        3×3 相机内参矩阵。
    dist : np.ndarray
        畸变系数。
    feature_type : str
        ``"sift"``（推荐）或 ``"orb"``。
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        feature_type: str = "sift",
    ) -> None:
        self.K = camera_matrix.astype(np.float64)
        self.dist = np.asarray(dist_coeffs, dtype=np.float64).ravel()

        # 特征提取器
        if feature_type == "sift":
            self._detector = cv2.SIFT_create(nfeatures=4000)
        elif feature_type == "orb":
            self._detector = cv2.ORB_create(nfeatures=4000)
        else:
            raise ValueError(f"不支持的特征类型: {feature_type}")
        self.feature_type = feature_type

        # FLANN 匹配器（SIFT）或 BF 匹配器（ORB）
        if feature_type == "sift":
            index_params = dict(algorithm=1, trees=5)  # KD-tree
            search_params = dict(checks=50)
            self._matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # 数据
        self.views: List[FeatureView] = []
        self.match_pairs: List[MatchPair] = []
        self._points3d: Dict[int, Tuple[float, float, float]] = {}
        self._next_point_id: int = 0

        # 地面 / 飞机分割结果
        self.ground_plane: Optional[Tuple[np.ndarray, float]] = None
        # (normal, d) where normal·X + d = 0
        self.ground_point_ids: Set[int] = set()
        self.aircraft_point_ids: Set[int] = set()

        self._timings: Dict[str, float] = {}

    # ==================================================================
    # 步骤 1 — 特征提取
    # ==================================================================

    def add_images(
        self, images: List[np.ndarray], names: Optional[List[str]] = None
    ) -> None:
        """添加图像并提取特征。"""
        if names is None:
            names = [f"view_{i:03d}" for i in range(len(images))]

        for name, img in zip(names, images):
            view = FeatureView(name, img)
            gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = self._detector.detectAndCompute(gray, None)
            view.keypoints = list(kp)
            view.descriptors = des
            view.point_ids = [-1] * len(kp)
            self.views.append(view)

        total_kp = sum(len(v.keypoints) for v in self.views)
        print(f"已添加 {len(images)} 张图像, 共 {total_kp} 个特征点 "
              f"(平均 {total_kp/len(images):.0f}/图)")

    # ==================================================================
    # 步骤 2 — 顺序匹配
    # ==================================================================

    def match_sequential(
        self,
        ratio_thresh: float = 0.75,
        min_matches: int = 30,
        cross_check_pairs: bool = True,
    ) -> int:
        """
        顺序匹配相邻图像对，并进行几何验证。

        Args:
            ratio_thresh: Lowe 比率检验阈值。
            min_matches: 几何验证后最少内点数。
            cross_check_pairs: 是否也匹配 i <-> i+2 对（增加连接性）。

        Returns:
            有效匹配对数量。
        """
        t0 = time.perf_counter()
        n = len(self.views)
        self.match_pairs = []

        for i in range(n - 1):
            self._match_pair(i, i + 1, ratio_thresh, min_matches)

        if cross_check_pairs:
            for i in range(n - 2):
                self._match_pair(i, i + 2, ratio_thresh, min_matches)

        n_valid = len(self.match_pairs)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"顺序匹配: {n_valid} 个有效对 "
              f"({n-1} 个连续对 + {n-2 if n>2 else 0} 个跳帧对) "
              f"耗时 {elapsed:.0f} ms")

        self._timings["match_sequential"] = elapsed
        return n_valid

    def _match_pair(
        self, idx_a: int, idx_b: int,
        ratio_thresh: float, min_matches: int,
    ) -> Optional[MatchPair]:
        """匹配一对图像并进行几何验证。"""
        va, vb = self.views[idx_a], self.views[idx_b]
        if va.descriptors is None or vb.descriptors is None:
            return None
        if len(va.keypoints) < 10 or len(vb.keypoints) < 10:
            return None

        # KNN 匹配 + 比率检验
        knn = self._matcher.knnMatch(va.descriptors, vb.descriptors, k=2)
        good = [m for m, n in knn if m.distance < ratio_thresh * n.distance]

        if len(good) < max(min_matches // 2, 10):
            return None

        # 提取匹配点坐标
        pts_a = np.array([va.keypoints[m.queryIdx].pt for m in good],
                         dtype=np.float64)
        pts_b = np.array([vb.keypoints[m.trainIdx].pt for m in good],
                         dtype=np.float64)

        # 基础矩阵 RANSAC
        F, inlier_mask = cv2.findFundamentalMat(
            pts_a, pts_b, cv2.FM_RANSAC, 3.0, 0.99)
        if F is None or inlier_mask.sum() < max(min_matches // 2, 15):
            return None

        inliers = inlier_mask.ravel().astype(bool)
        inlier_list = [good[i] for i in range(len(good)) if inliers[i]]

        # 本质矩阵
        E = self.K.T @ F @ self.K
        U, _, Vt = np.linalg.svd(E)
        S = np.array([1.0, 1.0, 0.0])
        E = U @ np.diag(S) @ Vt

        pair = MatchPair(idx_a, idx_b)
        pair.matches = good
        pair.inlier_matches = inlier_list
        pair.F = F
        pair.E = E
        self.match_pairs.append(pair)

        return pair

    # ==================================================================
    # 步骤 3 — 初始化
    # ==================================================================

    def initialize(
        self, pair_indices: Tuple[int, int] = (0, 1),
        min_inliers: int = 30,
    ) -> bool:
        """
        从一对视图初始化重建。

        Args:
            pair_indices: 用于初始化的视图索引。
            min_inliers: 最少内点数。

        Returns:
            成功返回 True。
        """
        t0 = time.perf_counter()
        i, j = pair_indices
        va, vb = self.views[i], self.views[j]

        # 查找该对的匹配关系
        pair = self._find_or_create_pair(i, j)
        if pair is None or len(pair.inlier_matches) < min_inliers:
            print(f"初始化失败: 视图 {i}<->{j} 内点数不足")
            return False

        # 提取内点
        pts_a = np.array([va.keypoints[m.queryIdx].pt
                          for m in pair.inlier_matches], dtype=np.float64)
        pts_b = np.array([vb.keypoints[m.trainIdx].pt
                          for m in pair.inlier_matches], dtype=np.float64)

        # 从本质矩阵恢复位姿
        n_pts, R_rel, t_rel, mask = cv2.recoverPose(
            pair.E, pts_a, pts_b, self.K)
        angle = np.linalg.norm(cv2.Rodrigues(R_rel)[0])
        print(f"初始化: 视图 {i} <-> {j} "
              f"({len(pts_a)} 个内点, 相对旋转 {np.rad2deg(angle):.1f}°)")

        # 设置相机 0 为世界原点
        va.R = np.eye(3)
        va.t = np.zeros((3, 1))
        va.registered = True
        vb.R = R_rel
        vb.t = t_rel.reshape(3, 1)
        vb.registered = True

        # 三角测量
        P0 = self.K @ np.hstack([va.R, va.t])
        P1 = self.K @ np.hstack([vb.R, vb.t])

        inlier_mask = mask.ravel().astype(bool)
        pts_a_in = pts_a[inlier_mask]
        pts_b_in = pts_b[inlier_mask]

        pts4d = cv2.triangulatePoints(P0, P1, pts_a_in.T, pts_b_in.T)
        pts3d = pts4d[:3] / pts4d[3]

        # 存储三维点
        inlier_indices = [i for i, ok in enumerate(inlier_mask) if ok]
        for k, (idx_3d, col) in enumerate(zip(inlier_indices, pts3d.T)):
            if col[2] > 0:
                pt_id = self._next_point_id
                self._next_point_id += 1
                self._points3d[pt_id] = (float(col[0]), float(col[1]), float(col[2]))
                # 关联关键点
                match = pair.inlier_matches[idx_3d]
                va.point_ids[match.queryIdx] = pt_id
                vb.point_ids[match.trainIdx] = pt_id

        n_pts = len(self._points3d)
        print(f"  三角测量得到 {n_pts} 个三维点")

        self._timings["initialize"] = (time.perf_counter() - t0) * 1000
        return n_pts >= 10

    def _find_or_create_pair(self, i: int, j: int) -> Optional[MatchPair]:
        """查找或创建一对视图的匹配。"""
        for pair in self.match_pairs:
            if (pair.idx_a == i and pair.idx_b == j):
                return pair
            if (pair.idx_a == j and pair.idx_b == i):
                return pair
        return self._match_pair(i, j, ratio_thresh=0.75, min_matches=30)

    # ==================================================================
    # 步骤 4 — 增量注册
    # ==================================================================

    def register_all_sequential(
        self, min_matches: int = 20, min_inliers: int = 15
    ) -> int:
        """
        按顺序使用本质矩阵注册所有未注册视图。

        对每个未注册视图：
          1. 找到与已注册视图的最佳匹配对
          2. 通过本质矩阵恢复相对位姿
          3. 从已注册邻居的绝对位姿推导新视图位姿
          4. 三角测量所有匹配特征
          5. 用新三维点更新已注册视图中的对应特征

        Returns:
            新注册的视图数量。
        """
        t0 = time.perf_counter()
        newly_registered = 0

        for vi in range(len(self.views)):
            view = self.views[vi]
            if view.registered:
                continue

            # 找到与已注册视图的最佳匹配对
            best_pair = None
            best_count = 0
            for pair in self.match_pairs:
                if pair.idx_a != vi and pair.idx_b != vi:
                    continue
                other = pair.idx_b if pair.idx_a == vi else pair.idx_a
                if not self.views[other].registered:
                    continue
                if len(pair.inlier_matches) > best_count:
                    best_count = len(pair.inlier_matches)
                    best_pair = pair

            if best_pair is None or best_count < min_matches:
                continue

            other_vi = (best_pair.idx_b if best_pair.idx_a == vi
                        else best_pair.idx_a)
            other_view = self.views[other_vi]

            # 提取匹配点对
            new_is_a = (best_pair.idx_a == vi)
            pts_new = []
            pts_other = []
            for m in best_pair.inlier_matches:
                qi = m.queryIdx if new_is_a else m.trainIdx
                ti = m.trainIdx if new_is_a else m.queryIdx
                pts_new.append(view.keypoints[qi].pt)
                pts_other.append(other_view.keypoints[ti].pt)

            if len(pts_new) < min_matches:
                continue

            pts_new_arr = np.array(pts_new, dtype=np.float64)
            pts_other_arr = np.array(pts_other, dtype=np.float64)

            # 本质矩阵 → 相对位姿
            E, inlier_mask = cv2.findEssentialMat(
                pts_new_arr, pts_other_arr, self.K,
                method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if E is None:
                continue

            inliers = inlier_mask.ravel().astype(bool)
            if inliers.sum() < min_inliers:
                continue

            n_pts, R_rel, t_rel, _ = cv2.recoverPose(
                E, pts_new_arr[inliers], pts_other_arr[inliers], self.K)

            # 新视图在世界坐标系中的位姿
            # X_other_cam = R_other * X_world + t_other
            # X_new_cam = R_rel * X_other_cam + t_rel
            #           = R_rel * R_other * X_world + R_rel * t_other + t_rel
            view.R = R_rel @ other_view.R
            view.t = (R_rel @ other_view.t.ravel() + t_rel.ravel()).reshape(3, 1)
            view.registered = True
            newly_registered += 1

            angle = np.linalg.norm(cv2.Rodrigues(R_rel)[0])
            print(f"  已注册 '{view.name}': {inliers.sum()} 内点, "
                  f"ΔR={np.rad2deg(angle):.0f}deg "
                  f"(via '{other_view.name}')")

            # 三角测量所有此对中的特征（不仅是内点，所有几何验证过的匹配）
            self._triangulate_pair(vi, other_vi, best_pair)

        elapsed = (time.perf_counter() - t0) * 1000
        n_total = sum(1 for v in self.views if v.registered)
        print(f"增量注册: {newly_registered} 个新视图 "
              f"(共 {n_total}/{len(self.views)} 个已注册) "
              f"耗时 {elapsed:.0f} ms")

        self._timings["register_all"] = elapsed
        return newly_registered

    def _triangulate_pair(
        self, vi_a: int, vi_b: int, pair: MatchPair
    ) -> int:
        """对一对视图的所有内点匹配进行三角测量。"""
        va, vb = self.views[vi_a], self.views[vi_b]
        if not va.registered or not vb.registered:
            return 0

        P_a = self.K @ np.hstack([va.R, va.t])
        P_b = self.K @ np.hstack([vb.R, vb.t])

        # 收集待三角测量的匹配
        to_triangulate = []
        for m in pair.inlier_matches:
            if pair.idx_a == vi_a:
                kp_a_idx, kp_b_idx = m.queryIdx, m.trainIdx
            else:
                kp_a_idx, kp_b_idx = m.trainIdx, m.queryIdx

            # 如果任一侧已有三维点，将对应关系传播给另一侧
            pt_a = va.point_ids[kp_a_idx]
            pt_b = vb.point_ids[kp_b_idx]
            if pt_a >= 0 and pt_b < 0:
                vb.point_ids[kp_b_idx] = pt_a
                continue
            if pt_b >= 0 and pt_a < 0:
                va.point_ids[kp_a_idx] = pt_b
                continue
            if pt_a >= 0 and pt_b >= 0:
                continue  # 都已关联

            to_triangulate.append((kp_a_idx, kp_b_idx))

        if not to_triangulate:
            return 0

        # 批量三角测量
        pts_a = np.array([va.keypoints[i].pt for i, _ in to_triangulate],
                         dtype=np.float64)
        pts_b = np.array([vb.keypoints[j].pt for _, j in to_triangulate],
                         dtype=np.float64)

        pts4d = cv2.triangulatePoints(P_a, P_b, pts_a.T, pts_b.T)
        pts3d = pts4d[:3] / pts4d[3]

        new_count = 0
        for k, ((kp_a_idx, kp_b_idx), col) in enumerate(
            zip(to_triangulate, pts3d.T)
        ):
            p = col
            # 前方检查
            pc_a = va.R @ p + va.t.ravel()
            pc_b = vb.R @ p + vb.t.ravel()
            if pc_a[2] <= 0 or pc_b[2] <= 0:
                continue

            pt_id = self._next_point_id
            self._next_point_id += 1
            self._points3d[pt_id] = (float(p[0]), float(p[1]), float(p[2]))
            va.point_ids[kp_a_idx] = pt_id
            vb.point_ids[kp_b_idx] = pt_id
            new_count += 1

        if new_count > 0:
            pass  # logged by caller if needed
        return new_count

    # ==================================================================
    # 步骤 5 — 捆绑调整
    # ==================================================================

    def bundle_adjust(
        self, iterations: int = 10, verbose: bool = True
    ) -> float:
        """
        基于重投影误差最小化的全局捆绑调整。

        使用交替优化: ① PnP 优化相机位姿 ② DLT 优化三维点。
        """
        t0 = time.perf_counter()
        reg_views = [v for v in self.views if v.registered]

        for it in range(iterations):
            # ① 优化相机位姿
            for view in reg_views:
                obj_pts_list = []
                img_pts_list = []
                for kp_idx, pt_id in enumerate(view.point_ids):
                    if pt_id >= 0 and pt_id in self._points3d:
                        kp = view.keypoints[kp_idx]
                        obj_pts_list.append(self._points3d[pt_id])
                        img_pts_list.append(kp.pt)

                if len(obj_pts_list) < 6:
                    continue

                obj_arr = np.array(obj_pts_list, dtype=np.float64)
                img_arr = np.array(img_pts_list, dtype=np.float64)

                ok, rvec, tvec = cv2.solvePnP(
                    obj_arr, img_arr, self.K, self.dist,
                    rvec=cv2.Rodrigues(view.R)[0], tvec=view.t,
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if ok:
                    view.R, _ = cv2.Rodrigues(rvec)
                    view.t = tvec.reshape(3, 1)

            # ② 优化三维点
            for pt_id in list(self._points3d.keys()):
                observing_views = []
                obs_pts = []
                for view in reg_views:
                    for kp_idx, pid in enumerate(view.point_ids):
                        if pid == pt_id:
                            kp = view.keypoints[kp_idx]
                            observing_views.append(view)
                            obs_pts.append(kp.pt)
                            break

                if len(observing_views) < 2:
                    continue

                # DLT 多视图三角测量
                A = np.zeros((2 * len(observing_views), 4))
                for i, (view, (u, v)) in enumerate(
                    zip(observing_views, obs_pts)
                ):
                    P = self.K @ np.hstack([view.R, view.t])
                    A[2*i] = u * P[2] - P[0]
                    A[2*i+1] = v * P[2] - P[1]

                _, _, Vt = np.linalg.svd(A, full_matrices=False)
                X = Vt[-1]
                p = X[:3] / X[3]

                # 检查在所有相机前方
                all_front = True
                for view in observing_views:
                    if (view.R @ p + view.t.ravel())[2] <= 0:
                        all_front = False
                        break

                if all_front:
                    self._points3d[pt_id] = (float(p[0]), float(p[1]), float(p[2]))

        err = self._compute_reproj_error()
        elapsed = (time.perf_counter() - t0) * 1000

        if verbose:
            n_v = len(reg_views)
            n_p = len(self._points3d)
            print(f"捆绑调整: {n_v} 视图, {n_p} 点 → "
                  f"重投影误差 = {err:.3f} px ({elapsed:.0f} ms)")

        self._timings["bundle_adjust"] = elapsed
        return err

    # ==================================================================
    # 步骤 6 — 地面平面拟合
    # ==================================================================

    def fit_ground_plane(
        self,
        distance_threshold: float = 0.02,
        min_inliers: int = 50,
        max_iterations: int = 500,
    ) -> Optional[np.ndarray]:
        """
        使用 RANSAC 从三维点云中拟合地面平面。

        假设：
          - 地面是最大的近似水平平面
          - 飞机点云位于地面上方

        Args:
            distance_threshold: 到平面的距离阈值（米）。
            min_inliers: 最少内点数。
            max_iterations: RANSAC 最大迭代次数。

        Returns:
            平面法向量 ``(3,)``，或 None。
        """
        if len(self._points3d) < min_inliers:
            print(f"三维点太少 ({len(self._points3d)} < {min_inliers})")
            return None

        pts = np.array(list(self._points3d.values()), dtype=np.float64)
        pt_ids = list(self._points3d.keys())

        best_inliers: Set[int] = set()
        best_normal = np.array([0.0, 0.0, 1.0])
        best_d = 0.0

        rng = np.random.default_rng(42)

        for _ in range(max_iterations):
            # 随机选 3 个点
            sample_idx = rng.choice(len(pts), 3, replace=False)
            p0, p1, p2 = pts[sample_idx]
            normal = np.cross(p1 - p0, p2 - p0)
            n_norm = np.linalg.norm(normal)
            if n_norm < 1e-9:
                continue
            normal = normal / n_norm
            d = -np.dot(normal, p0)

            # 计数内点
            distances = np.abs(pts @ normal + d)
            inlier_mask = distances < distance_threshold
            inlier_set = {pt_ids[i] for i in np.where(inlier_mask)[0]}

            if len(inlier_set) > len(best_inliers):
                best_inliers = inlier_set
                best_normal = normal
                best_d = d

        if len(best_inliers) < min_inliers:
            print(f"地面拟合: 内点不足 ({len(best_inliers)} < {min_inliers})")
            return None

        # 确保法向量朝上
        if best_normal[2] < 0:
            best_normal = -best_normal
            best_d = -best_d

        self.ground_plane = (best_normal, best_d)
        self.ground_point_ids = best_inliers

        print(f"地面平面: 法向量=({best_normal[0]:.3f}, {best_normal[1]:.3f}, "
              f"{best_normal[2]:.3f}), {len(best_inliers)} 个内点")
        return best_normal

    # ==================================================================
    # 步骤 7 — 飞机点云分割
    # ==================================================================

    def segment_aircraft(
        self,
        height_min: float = 0.01,
        height_max: float = 0.50,
        cluster_eps: float = 0.03,
        min_cluster_size: int = 20,
    ) -> Set[int]:
        """
        从三维点云中分割出飞机点（位于地面上方的点）。

        策略：
          1. 按地面高度过滤：地面上方 height_min ~ height_max 的点
          2. DBSCAN 聚类：去除离散噪声点
          3. 选取最大的非地面聚类作为飞机

        Args:
            height_min: 飞机点距离地面的最小高度（米）。
            height_max: 飞机点距离地面的最大高度（米）。
            cluster_eps: DBSCAN 聚类半径（米）。
            min_cluster_size: 最小聚类大小。

        Returns:
            飞机点的 point_id 集合。
        """
        if self.ground_plane is None:
            print("请先运行 fit_ground_plane()")
            return set()

        normal, d = self.ground_plane

        # 按高度过滤
        candidate_ids = []
        candidate_pts = []
        for pt_id, (x, y, z) in self._points3d.items():
            if pt_id in self.ground_point_ids:
                continue  # 已标记为地面
            p = np.array([x, y, z])
            height = np.dot(normal, p) + d  # 到地面平面的有符号距离
            if height_min < height < height_max:
                candidate_ids.append(pt_id)
                candidate_pts.append(p)

        if len(candidate_pts) < min_cluster_size:
            print(f"候选飞机点太少 ({len(candidate_pts)} < {min_cluster_size})")
            return set()

        pts_arr = np.array(candidate_pts, dtype=np.float64)

        # 简单聚类：选取最大的连通分量
        # （用基于距离的简单贪心聚类代替 DBSCAN，避免依赖 sklearn）
        labels = self._simple_cluster(pts_arr, cluster_eps)

        # 选取最大的聚类
        unique_labels, counts = np.unique(labels, return_counts=True)
        best_label = unique_labels[np.argmax(counts)]
        cluster_mask = labels == best_label

        if cluster_mask.sum() < min_cluster_size:
            print(f"最大聚类太小 ({cluster_mask.sum()} < {min_cluster_size})")
            return set()

        self.aircraft_point_ids = {
            candidate_ids[i] for i in np.where(cluster_mask)[0]
        }

        # 计算飞机质心和大致尺寸
        ac_pts = pts_arr[cluster_mask]
        centroid = ac_pts.mean(axis=0)
        extent = ac_pts.max(axis=0) - ac_pts.min(axis=0)

        print(f"飞机点云: {len(self.aircraft_point_ids)} 个点, "
              f"质心=({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}), "
              f"尺寸=({extent[0]:.3f}, {extent[1]:.3f}, {extent[2]:.3f}) m")

        return self.aircraft_point_ids

    @staticmethod
    def _simple_cluster(pts: np.ndarray, eps: float) -> np.ndarray:
        """简易距离聚类（替代 DBSCAN，无外部依赖）。"""
        n = len(pts)
        labels = np.full(n, -1, dtype=int)
        current_label = 0

        for i in range(n):
            if labels[i] >= 0:
                continue
            # BFS 从点 i 开始
            labels[i] = current_label
            queue = [i]
            while queue:
                cur = queue.pop(0)
                dists = np.linalg.norm(pts - pts[cur], axis=1)
                neighbors = np.where((dists < eps) & (labels < 0))[0]
                for nb in neighbors:
                    labels[nb] = current_label
                    queue.append(nb)
            current_label += 1

        return labels

    # ==================================================================
    # 飞机姿态估计
    # ==================================================================

    def estimate_aircraft_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        从分割出的飞机点云中估计飞机姿态。

        Returns:
            ``(R, t)`` 将飞机局部坐标系变换到世界坐标系。
            飞机局部坐标系: 原点 = 点云质心, Z = 点云主方向（最长轴）。
        """
        if not self.aircraft_point_ids:
            print("请先运行 segment_aircraft()")
            return None

        pts = np.array([
            self._points3d[pt_id] for pt_id in self.aircraft_point_ids
        ], dtype=np.float64)

        centroid = pts.mean(axis=0)
        pts_c = pts - centroid

        # PCA: 主方向
        _, _, Vt = np.linalg.svd(pts_c, full_matrices=False)
        # 第一主成分 = 最大方差方向 (飞机前后轴)
        pca_x = Vt[0]
        pca_y = Vt[1]
        pca_z = np.cross(pca_x, pca_y)

        R = np.column_stack([pca_x, pca_y, pca_z])
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1

        # Euler 角
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6
        if not singular:
            rx = math.degrees(math.atan2(R[2, 1], R[2, 2]))
            ry = math.degrees(math.atan2(-R[2, 0], sy))
            rz = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        else:
            rx = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
            ry = math.degrees(math.atan2(-R[2, 0], sy))
            rz = 0.0

        print(f"飞机姿态 (PCA):")
        print(f"  位置 (m):     ({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f})")
        print(f"  欧拉角 (°):   roll={rx:.2f}  pitch={ry:.2f}  yaw={rz:.2f}")

        return R, centroid

    # ==================================================================
    # 辅助
    # ==================================================================

    def _compute_reproj_error(self) -> float:
        total = 0.0
        n = 0
        for view in self.views:
            if not view.registered:
                continue
            for kp_idx, pt_id in enumerate(view.point_ids):
                if pt_id < 0 or pt_id not in self._points3d:
                    continue
                p3d = np.array(self._points3d[pt_id])
                pc = view.R @ p3d + view.t.ravel()
                if pc[2] <= 0:
                    continue
                up = pc[0]/pc[2]*self.K[0,0] + self.K[0,2]
                vp = pc[1]/pc[2]*self.K[1,1] + self.K[1,2]
                u, v = view.keypoints[kp_idx].pt
                total += math.hypot(up-u, vp-v)
                n += 1
        return total / n if n > 0 else float("inf")

    @property
    def points_3d(self) -> Dict[int, Tuple[float, float, float]]:
        return dict(self._points3d)

    def summary(self) -> str:
        n_v = sum(1 for v in self.views if v.registered)
        n_p = len(self._points3d)
        err = self._compute_reproj_error()
        lines = [
            f"特征 SfM 摘要",
            f"  视图:     {n_v}/{len(self.views)} 已注册",
            f"  三维点:   {n_p}",
            f"  重投影误差: {err:.3f} px",
            f"  地面内点: {len(self.ground_point_ids)}",
            f"  飞机点:   {len(self.aircraft_point_ids)}",
        ]
        for k, v in self._timings.items():
            lines.append(f"  {k}: {v:.0f} ms")
        return "\n".join(lines)


# ===================================================================
# 测试：在单张图上提取特征并可视化
# ===================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

    img_path = sys.argv[1] if len(sys.argv) > 1 else "data/plane.png"
    print(f"特征 SfM — 测试特征提取: {img_path}")

    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取: {img_path}")
        sys.exit(1)

    # 提取特征
    K = np.array([[1200, 0, img.shape[1]/2],
                   [0, 1200, img.shape[0]/2],
                   [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5)

    sfm = FeatureBasedSfM(K, dist, feature_type="sift")
    sfm.add_images([img])

    view = sfm.views[0]
    print(f"特征点: {len(view.keypoints)}")

    # 可视化
    vis = cv2.drawKeypoints(img, view.keypoints, None,
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    out = "output/feature_test.png"
    import os
    os.makedirs("output", exist_ok=True)
    cv2.imwrite(out, vis)
    print(f"特征可视化: {out}")

    # 统计
    sizes = [kp.size for kp in view.keypoints]
    print(f"关键点尺度: min={min(sizes):.1f}, median={np.median(sizes):.1f}, max={max(sizes):.1f}")
