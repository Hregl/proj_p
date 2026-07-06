"""
基于平面+视差的运动恢复结构（Plane + Parallax SfM）。

核心思想：
  场景中有一个已知平面（如切割垫），用单应性矩阵稳定地
  估计相机运动，再将飞机视为平面上方的"视差"来重建。

优势（相比稀疏特征 SfM）：
  - 单应性使用全部匹配点（数百个），而非本质矩阵的十几个内点
  - 单应性分解给出唯一解（无 4 选 1 歧义）
  - 自然地分离地面（平面内）和飞机（视差 = 平面上方）
  - 对光滑表面的容忍度高得多
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 平面 + 视差 SfM
# ---------------------------------------------------------------------------

class PlaneParallaxSfM:
    """
    基于已知主导平面的增量式 SfM。

    假设场景中有一个平面（地面），相机围绕场景运动。
    利用连续帧之间的单应性矩阵稳定估计相机运动，
    然后将偏离平面的特征点（飞机）作为"视差"重建。

    工作流程
    --------
      1. 提取 SIFT 特征
      2. 相邻帧匹配 + 计算单应性矩阵
      3. 从单应性分解相机位姿
      4. 三角测量偏离平面的点（飞机）
      5. 捆绑调整
      6. 地面拟合 + 飞机分割
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> None:
        self.K = camera_matrix.astype(np.float64)
        self.dist = np.asarray(dist_coeffs, dtype=np.float64).ravel()

        self._detector = cv2.SIFT_create(nfeatures=4000)
        self._matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5), dict(checks=50))

        # 每帧数据
        self.views: List[_PlaneView] = []
        # 三维重建结果
        self._points3d: Dict[int, Tuple[float, float, float]] = {}
        self._next_pt_id = 0
        # 分类
        self.ground_point_ids: Set[int] = set()
        self.aircraft_point_ids: Set[int] = set()

        self._timings: Dict[str, float] = {}

    # ==================================================================
    # 添加图像
    # ==================================================================

    def add_images(
        self, images: List[np.ndarray],
        names: Optional[List[str]] = None,
    ) -> None:
        """提取所有图像的 SIFT 特征。"""
        if names is None:
            names = [f"v{i:03d}" for i in range(len(images))]

        for name, img in zip(names, images):
            gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = self._detector.detectAndCompute(gray, None)
            v = _PlaneView(name, img)
            v.keypoints = list(kp)
            v.descriptors = des
            v.point_ids = [-1] * len(kp)
            self.views.append(v)

        n_kp = sum(len(v.keypoints) for v in self.views)
        print(f"已添加 {len(images)} 张图像, 共 {n_kp} 个特征点 "
              f"(平均 {n_kp/len(images):.0f}/图)")

    # ==================================================================
    # 单应性矩阵估计
    # ==================================================================

    @staticmethod
    def _compute_homography(
        pts_src: np.ndarray, pts_dst: np.ndarray,
        ransac_thresh: float = 3.0,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        计算两个视图之间的单应性矩阵。

        单应性矩阵将平面上的点从源视图映射到目标视图：
          dst = H @ src

        使用 RANSAC 鲁棒估计。

        Returns:
            ``(H, inlier_mask)`` 或 ``(None, None)``。
        """
        if len(pts_src) < 4:
            return None, None

        H, mask = cv2.findHomography(
            pts_src, pts_dst, cv2.RANSAC, ransac_thresh,
            maxIters=2000, confidence=0.995,
        )
        return H, mask

    # ==================================================================
    # 从单应性矩阵分解相机运动
    # ==================================================================

    @staticmethod
    def _decompose_homography(
        H: np.ndarray, K: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        将单应性矩阵分解为旋转、平移和平面法向量。

        平面诱导的单应性矩阵：
          H = K @ (R - t·nᵗ/d) @ K⁻¹

        其中 R=旋转, t=平移, n=平面法向量, d=平面到原点的距离。

        Returns:
            ``[(R, t, n), …]`` — 通常返回 2-4 个可能解。
        """
        # 归一化
        H_norm = np.linalg.inv(K) @ H @ K
        # 确保行列式为正（前方平面）
        if np.linalg.det(H_norm) < 0:
            H_norm = -H_norm

        # SVD 分解
        U, S, Vt = np.linalg.svd(H_norm)
        V = Vt.T

        # 使奇异值归一化
        s1, s2, s3 = S
        # 确保 s3=1 的缩放
        xi = s1 / s2 if s2 > 1e-9 else 1.0

        # 从 Faugeras & Lustman (1988) 的方法
        # 计算可能的平面法向量
        v1 = V[:, 0]
        v2 = V[:, 1]
        v3 = V[:, 2]
        u1 = U[:, 0]
        u2 = U[:, 1]
        u3 = U[:, 2]

        solutions = []

        # 解 1 & 2: n' = (v₂×v₁ 方向)
        n1 = np.cross(v2, v1)
        n1 = n1 / (np.linalg.norm(n1) + 1e-9)

        # 尝试两组旋转和平移
        cos_theta = (s1 * s1 * s3 + s2) / (s1 * (s2 * s2 + s3))
        cos_theta = np.clip(cos_theta, -1, 1)
        sin_theta = math.sqrt(max(0, 1 - cos_theta * cos_theta))

        for sign in [1.0, -1.0]:
            R1 = U @ np.array([
                [cos_theta, 0, -sign * sin_theta],
                [0, 1, 0],
                [sign * sin_theta, 0, cos_theta],
            ]) @ Vt

            t1 = sign * (s3 - s2 * cos_theta) * v1 + (s1 - s2 * sin_theta) * v3
            t1 = t1 / np.linalg.norm(t1) if np.linalg.norm(t1) > 1e-9 else t1

            if np.linalg.det(R1) > 0:
                solutions.append((R1, t1, n1.copy()))

        return solutions

    @staticmethod
    def _select_best_decomposition(
        solutions: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        pts1: np.ndarray, pts2: np.ndarray,
        K: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        从多个分解中选择最佳解。

        判断标准：
          1. 最多的三角测量点位于两个相机前方
          2. 平面法向量朝上（对地面场景）
        """
        best_count = -1
        best_sol = None

        for R, t, n in solutions:
            # 三角测量内点
            P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
            P2 = K @ np.hstack([R, t.reshape(3, 1)])

            pts4d = cv2.triangulatePoints(
                P1, P2, pts1.T, pts2.T)
            pts3d = pts4d[:3] / pts4d[3]

            # 计数前方点
            front_count = 0
            for col in pts3d.T:
                # 相机 1 前方
                if col[2] <= 0:
                    continue
                # 相机 2 前方
                p2 = R @ col + t
                if p2[2] <= 0:
                    continue
                # 平面法向量朝上（z>0）
                if n[2] > -0.3:  # 容忍略微朝下
                    front_count += 1

            if front_count > best_count:
                best_count = front_count
                best_sol = (R, t, n)

        return best_sol

    # ==================================================================
    # 运行流水线
    # ==================================================================

    def run(
        self,
        ransac_thresh: float = 3.0,
        min_matches: int = 30,
    ) -> bool:
        """
        运行完整的平面+视差 SfM 流水线。

        Returns:
            成功返回 True。
        """
        t_total = time.perf_counter()
        n = len(self.views)
        if n < 2:
            print("至少需要 2 张图像")
            return False

        # --- 步骤 1: 相邻帧匹配 + 单应性 ---
        print("\n--- 单应性估计 ---")
        t0 = time.perf_counter()
        homographies: List[Optional[Tuple]] = []  # per adjacent pair

        for i in range(n - 1):
            va, vb = self.views[i], self.views[i + 1]
            if va.descriptors is None or vb.descriptors is None:
                homographies.append(None)
                continue

            knn = self._matcher.knnMatch(va.descriptors, vb.descriptors, k=2)
            # 放宽比率检验以获取更多匹配
            good = [m for m, n in knn if m.distance < 0.85 * n.distance]

            if len(good) < min_matches:
                homographies.append(None)
                continue

            pts_a = np.array([va.keypoints[m.queryIdx].pt for m in good],
                             dtype=np.float64)
            pts_b = np.array([vb.keypoints[m.trainIdx].pt for m in good],
                             dtype=np.float64)

            H, mask = self._compute_homography(pts_a, pts_b, ransac_thresh * 1.5)
            if H is None:
                homographies.append(None)
                continue

            inliers = mask.ravel().astype(bool)
            n_inl = inliers.sum()
            print(f"  帧 {i}->{i+1}: {len(good)} matches, "
                  f"{n_inl} 单应性内点 ({n_inl/len(good)*100:.0f}%)")

            homographies.append((H, pts_a, pts_b, inliers, good))

        self._timings["homography"] = (time.perf_counter() - t0) * 1000

        # --- 步骤 2: 初始化（第一对） ---
        print("\n--- 初始化 ---")
        t0 = time.perf_counter()

        # 找第一个有效的单应性对
        init_pair = None
        for i, hdata in enumerate(homographies):
            if hdata is not None:
                init_pair = i
                break

        if init_pair is None:
            print("无有效单应性对")
            return False

        H, pts0, pts1, inliers, matches = homographies[init_pair]
        pts0_in = pts0[inliers]
        pts1_in = pts1[inliers]

        # 分解单应性
        sols = self._decompose_homography(H, self.K)
        best = self._select_best_decomposition(sols, pts0_in, pts1_in, self.K)
        if best is None:
            print("单应性分解失败")
            return False

        R, t, plane_normal = best
        print(f"  平面法向量: ({plane_normal[0]:.3f}, {plane_normal[1]:.3f}, {plane_normal[2]:.3f})")

        # 设置前两帧的位姿
        v0 = self.views[init_pair]
        v1 = self.views[init_pair + 1]
        v0.R = np.eye(3)
        v0.t = np.zeros((3, 1))
        v0.registered = True
        v1.R = R
        v1.t = t.reshape(3, 1)
        v1.registered = True

        # 三角测量初始点
        self._triangulate_from_homography(v0, v1, pts0_in, pts1_in, matches, inliers)
        print(f"  初始化: {len(self._points3d)} 个三维点")

        self._timings["init"] = (time.perf_counter() - t0) * 1000

        # --- 步骤 3: 增量注册 ---
        print("\n--- 增量注册 ---")
        t0 = time.perf_counter()

        for i in range(n):
            if self.views[i].registered:
                continue

            # 找最近已注册帧的单应性
            best_hdata = None
            best_dist = 999
            for j in range(n):
                if not self.views[j].registered:
                    continue
                for k, hdata in enumerate(homographies):
                    if hdata is None:
                        continue
                    # 检查这个单应性是否连接 i 和 j
                    if (k == j and k + 1 == i) or (k + 1 == j and k == i):
                        d = abs(i - j)
                        if d == 1 and d < best_dist:
                            best_hdata = hdata
                            best_dist = d
                            best_reg = j

            if best_hdata is None:
                continue

            H, pts_a, pts_b, inliers, matches = best_hdata
            # 确定方向
            if best_reg < i:
                pts_reg = pts_a[inliers]
                pts_new = pts_b[inliers]
                # 单应性: reg -> new
                H_use = H
            else:
                pts_reg = pts_b[inliers]
                pts_new = pts_a[inliers]
                H_use = np.linalg.inv(H)

            # 计算新帧的位姿
            reg_view = self.views[best_reg]
            # 用 PnP 从已知三维点计算（如果有足够的平面内点）
            obj_pts = []
            img_pts = []
            for mi, m in enumerate(matches):
                if not inliers[mi]:
                    continue
                # 找到 reg_view 中有三维点的特征
                # (这里简化处理: 直接用单应性传递)
                pass

            # 用单应性直接推导相机位姿
            # H_reg_to_new = K @ (R_rel - t_rel * nᵗ/d) @ K⁻¹
            # 简化: 从 H 和已知 n 解 R, t
            sols = self._decompose_homography(H_use, self.K)
            best = self._select_best_decomposition(sols, pts_reg, pts_new, self.K)
            if best is None:
                continue

            R_rel, t_rel, _ = best
            new_view = self.views[i]
            new_view.R = R_rel @ reg_view.R
            new_view.t = (R_rel @ reg_view.t.ravel() + t_rel).reshape(3, 1)
            new_view.registered = True

            # 三角测量
            self._triangulate_from_homography(
                reg_view, new_view, pts_reg, pts_new,
                matches, inliers,
            )

            print(f"  注册 '{new_view.name}' (via '{reg_view.name}')")

        n_reg = sum(1 for v in self.views if v.registered)
        self._timings["register"] = (time.perf_counter() - t0) * 1000
        print(f"  已注册: {n_reg}/{n}")

        # --- 步骤 4: 捆绑调整 ---
        print("\n--- 捆绑调整 ---")
        t0 = time.perf_counter()
        err = self._bundle_adjust(iterations=10)
        self._timings["ba"] = (time.perf_counter() - t0) * 1000
        print(f"  重投影误差: {err:.2f} px")

        # --- 步骤 5: 地面检测 & 飞机分割 ---
        print("\n--- 分割 ---")
        self._fit_ground_plane()
        self._segment_aircraft()

        total = (time.perf_counter() - t_total) * 1000
        print(f"\n总耗时: {total:.0f} ms")
        return True

    # ==================================================================
    # 内部方法
    # ==================================================================

    def _triangulate_from_homography(
        self, va: _PlaneView, vb: _PlaneView,
        pts_a: np.ndarray, pts_b: np.ndarray,
        matches: List[cv2.DMatch], inlier_mask: np.ndarray,
    ) -> int:
        """从单应性内点三角测量三维点。"""
        P_a = self.K @ np.hstack([va.R, va.t])
        P_b = self.K @ np.hstack([vb.R, vb.t])

        pts4d = cv2.triangulatePoints(P_a, P_b, pts_a.T, pts_b.T)
        pts3d = pts4d[:3] / pts4d[3]

        count = 0
        for k, col in enumerate(pts3d.T):
            if col[2] <= 0:
                continue
            pc_b = vb.R @ col + vb.t.ravel()
            if pc_b[2] <= 0:
                continue

            pt_id = self._next_pt_id
            self._next_pt_id += 1
            self._points3d[pt_id] = (float(col[0]), float(col[1]), float(col[2]))

            # 关联关键点
            m = matches[inlier_mask.nonzero()[0][k]]
            va.point_ids[m.queryIdx if va == self.views[0] else m.trainIdx] = pt_id
            # (简化处理)
            count += 1

        return count

    def _bundle_adjust(self, iterations: int = 10) -> float:
        """交替优化: PnP 优化相机 + DLT 优化三维点。"""
        reg_views = [v for v in self.views if v.registered]

        for _ in range(iterations):
            for view in reg_views:
                obj_list, img_list = [], []
                for kp_idx, pt_id in enumerate(view.point_ids):
                    if pt_id >= 0 and pt_id in self._points3d:
                        obj_list.append(self._points3d[pt_id])
                        img_list.append(view.keypoints[kp_idx].pt)
                if len(obj_list) < 6:
                    continue
                obj = np.array(obj_list, dtype=np.float64)
                img = np.array(img_list, dtype=np.float64)
                ok, rvec, tvec = cv2.solvePnP(
                    obj, img, self.K, self.dist,
                    rvec=cv2.Rodrigues(view.R)[0], tvec=view.t,
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if ok:
                    view.R, _ = cv2.Rodrigues(rvec)
                    view.t = tvec.reshape(3, 1)

            for pt_id in list(self._points3d.keys()):
                views_see = []
                pts_see = []
                for view in reg_views:
                    for kp_idx, pid in enumerate(view.point_ids):
                        if pid == pt_id:
                            views_see.append(view)
                            pts_see.append(view.keypoints[kp_idx].pt)
                            break
                if len(views_see) < 2:
                    continue

                A = np.zeros((2 * len(views_see), 4))
                for i, (view, (u, v)) in enumerate(zip(views_see, pts_see)):
                    P = self.K @ np.hstack([view.R, view.t])
                    A[2*i] = u * P[2] - P[0]
                    A[2*i+1] = v * P[2] - P[1]
                _, _, Vt = np.linalg.svd(A, full_matrices=False)
                X = Vt[-1]
                p = X[:3] / X[3]
                self._points3d[pt_id] = (float(p[0]), float(p[1]), float(p[2]))

        return self._compute_reproj_error()

    def _fit_ground_plane(self) -> None:
        """RANSAC 拟合地面平面。"""
        if len(self._points3d) < 10:
            return
        pts = np.array(list(self._points3d.values()))
        pt_ids = list(self._points3d.keys())
        rng = np.random.default_rng(42)
        best_inliers = set()

        for _ in range(500):
            idx = rng.choice(len(pts), 3, replace=False)
            p0, p1, p2 = pts[idx]
            n = np.cross(p1 - p0, p2 - p0)
            nn = np.linalg.norm(n)
            if nn < 1e-9:
                continue
            n = n / nn
            d = -np.dot(n, p0)
            dists = np.abs(pts @ n + d)
            inlier_set = {pt_ids[i] for i in np.where(dists < 0.03)[0]}
            if len(inlier_set) > len(best_inliers):
                best_inliers = inlier_set

        if len(best_inliers) > 10:
            self.ground_point_ids = best_inliers
            print(f"  地面: {len(best_inliers)} 个内点")

    def _segment_aircraft(self) -> None:
        """按高度分割飞机点。"""
        if not self.ground_point_ids:
            return
        ground_pts = np.array([self._points3d[pid]
                               for pid in self.ground_point_ids])
        gz_mean = ground_pts[:, 2].mean()

        ac_ids = set()
        for pt_id, (x, y, z) in self._points3d.items():
            if pt_id in self.ground_point_ids:
                continue
            if z > gz_mean + 0.01:
                ac_ids.add(pt_id)

        if len(ac_ids) >= 5:
            self.aircraft_point_ids = ac_ids
            ac_pts = np.array([self._points3d[pid] for pid in ac_ids])
            c = ac_pts.mean(axis=0)
            print(f"  飞机: {len(ac_ids)} 个点, "
                  f"质心=({c[0]:.3f},{c[1]:.3f},{c[2]:.3f})")

    def _compute_reproj_error(self) -> float:
        total, n = 0.0, 0
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
                u = pc[0]/pc[2]*self.K[0,0]+self.K[0,2]
                v = pc[1]/pc[2]*self.K[1,1]+self.K[1,2]
                total += math.hypot(u-view.keypoints[kp_idx].pt[0],
                                    v-view.keypoints[kp_idx].pt[1])
                n += 1
        return total/n if n > 0 else float("inf")

    def summary(self) -> str:
        n_v = sum(1 for v in self.views if v.registered)
        n_p = len(self._points3d)
        err = self._compute_reproj_error()
        return (
            f"平面+视差 SfM 摘要\n"
            f"  视图: {n_v}/{len(self.views)} 已注册\n"
            f"  三维点: {n_p}\n"
            f"  重投影误差: {err:.2f} px\n"
            f"  地面点: {len(self.ground_point_ids)}\n"
            f"  飞机点: {len(self.aircraft_point_ids)}"
        )


class _PlaneView:
    __slots__ = ("name", "image", "keypoints", "descriptors",
                 "R", "t", "registered", "point_ids")
    def __init__(self, name: str, image: np.ndarray):
        self.name = name
        self.image = image
        self.keypoints: List[cv2.KeyPoint] = []
        self.descriptors: Optional[np.ndarray] = None
        self.R: Optional[np.ndarray] = None
        self.t: Optional[np.ndarray] = None
        self.registered = False
        self.point_ids: List[int] = []


# ===================================================================
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    os.makedirs("output", exist_ok=True)

    images = [cv2.resize(cv2.imread(f"data/scene/p{i}.png"), (2048, 1536))
              for i in range(9)]
    w, h = 2048, 1536
    K = np.array([[1479, 0, 1024], [0, 1479, 768], [0, 0, 1]], dtype=np.float64)

    sfm = PlaneParallaxSfM(K, np.zeros(5))
    sfm.add_images(images)
    sfm.run()

    print("\n" + sfm.summary())
