"""
基于三维模型的姿态跟踪（Model-Based Pose Estimation）。

利用飞机的精准三维模型，通过"渲染-匹配-PnP"的方式估计
飞机在真实照片中的六自由度姿态。无需在飞机上贴任何标记。

核心流程
--------
  1. 加载飞机三维模型（OBJ 文件或内置简易模型）
  2. 从球面分布的虚拟视角渲染模型 → 提取 SIFT 特征
  3. 建立特征数据库：{描述子 → 三维坐标}
  4. 对真实照片：提取 SIFT → 匹配数据库 → PnP → 姿态
  5. 结合地面参考（切割垫网格）→ 飞机在世界系中的姿态

优势
----
  - 飞机上不需要 ArUco 标记或白色圆点
  - 三维模型提供任意密度的"虚拟标记点"
  - 天然处理遮挡和视角变化
"""

from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from scipy.spatial import KDTree


# ===================================================================
# 三维模型接口
# ===================================================================


class Model3D:
    """
    简易三维模型（三角面片网格）。

    支持两种来源：
      1. 从 OBJ 文件加载
      2. 使用内置的 AircraftModel（简易飞机几何体）
    """

    def __init__(self) -> None:
        self.vertices: List[np.ndarray] = []   # (N, 3)
        self.faces: List[Tuple[int, int, int]] = []  # 顶点索引三元组
        self.name: str = "unnamed"

    @staticmethod
    def from_obj(filepath: str) -> "Model3D":
        """从 Wavefront OBJ 文件加载模型。"""
        model = Model3D()
        model.name = os.path.basename(filepath)

        with open(filepath, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == "v":
                    model.vertices.append(np.array(
                        [float(parts[1]), float(parts[2]), float(parts[3])]
                    ))
                elif parts[0] == "f":
                    # 处理 v/vt/vn 格式，三角剖分
                    indices = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                    if len(indices) == 3:
                        model.faces.append(tuple(indices))
                    elif len(indices) == 4:
                        model.faces.append((indices[0], indices[1], indices[2]))
                        model.faces.append((indices[0], indices[2], indices[3]))

        print(f"加载模型 '{model.name}': "
              f"{len(model.vertices)} 顶点, {len(model.faces)} 三角面")
        return model

    @staticmethod
    def from_aircraft_model() -> "Model3D":
        """从内置的 AircraftModel 构建。"""
        from sls_calib.scene_renderer import AircraftModel

        model = Model3D()
        model.name = "aircraft_builtin"

        ac = AircraftModel()
        vert_map: Dict[Tuple[float, float, float], int] = {}
        vertex_list: List[np.ndarray] = []

        for face in ac.faces:
            face_indices = []
            for v in face.vertices:
                key = (round(v[0], 6), round(v[1], 6), round(v[2], 6))
                if key not in vert_map:
                    vert_map[key] = len(vertex_list)
                    vertex_list.append(v)
                face_indices.append(vert_map[key])
            # 三角剖分多边形面（四边形→2个三角形）
            if len(face_indices) == 3:
                model.faces.append(tuple(face_indices))
            elif len(face_indices) == 4:
                model.faces.append((face_indices[0], face_indices[1], face_indices[2]))
                model.faces.append((face_indices[0], face_indices[2], face_indices[3]))

        model.vertices = vertex_list
        print(f"内置飞机模型: {len(model.vertices)} 顶点, {len(model.faces)} 三角面")
        return model

    def sample_surface_points(self, n_points: int = 500) -> np.ndarray:
        """
        在模型表面均匀采样三维点。

        Returns:
            ``(n_points, 3)`` 采样点坐标。
        """
        if not self.faces:
            return np.zeros((0, 3))

        # 按面积加权采样面
        areas = []
        face_verts = []
        for f in self.faces:
            v0, v1, v2 = [self.vertices[i] for i in f]
            area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            areas.append(area)
            face_verts.append((v0, v1, v2))

        total_area = sum(areas)
        probs = np.array(areas) / total_area

        rng = np.random.default_rng(42)
        face_indices = rng.choice(len(self.faces), n_points, p=probs)

        points = []
        for fi in face_indices:
            v0, v1, v2 = face_verts[fi]
            r1 = math.sqrt(rng.random())
            r2 = rng.random()
            p = (1 - r1) * v0 + r1 * (1 - r2) * v1 + r1 * r2 * v2
            points.append(p)

        return np.array(points, dtype=np.float64)


# ===================================================================
# 基于模型的姿态估计器
# ===================================================================


class ModelBasedTracker:
    """
    利用三维模型估计物体在图像中的姿态。

    使用方法
    --------
        tracker = ModelBasedTracker(K, model, marker_size=0.03)
        tracker.build_database(n_views=100)
        R, t, inliers = tracker.estimate_pose(image)
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        model: Model3D,
        marker_size: float = 0.03,
        feature_type: str = "sift",
    ) -> None:
        self.K = camera_matrix.astype(np.float64)
        self.model = model
        self.marker_size = marker_size
        self.feature_type = feature_type

        # 特征提取器
        if feature_type == "sift":
            self._detector = cv2.SIFT_create(nfeatures=2000)
            self._matcher = cv2.FlannBasedMatcher(
                dict(algorithm=1, trees=5), dict(checks=50))
        else:
            self._detector = cv2.ORB_create(nfeatures=2000)
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        # 特征数据库
        self._db_descriptors: Optional[np.ndarray] = None  # (M, 128)
        self._db_points3d: Optional[np.ndarray] = None     # (M, 3)
        self._db_kdtree: Optional[KDTree] = None

    # ==================================================================
    # 构建特征数据库
    # ==================================================================

    def build_database(
        self,
        n_views: int = 100,
        distance: float = 0.5,
        model_scale: float = 1.0,
        debug: bool = False,
    ) -> int:
        """
        从球面分布的虚拟视角渲染模型并建立特征数据库。

        对每个视角：
          1. 渲染模型（简易线框 + 着色面）
          2. 提取 SIFT 特征
          3. 反向投影计算每个特征的 3D 坐标

        Args:
            n_views: 虚拟视角数量（越多越密，推荐 100-200）。
            distance: 虚拟相机到模型中心的距离（米）。
            model_scale: 模型缩放因子。
            debug: 保存渲染视图以供检查。

        Returns:
            数据库中特征点总数。
        """
        t0 = time.perf_counter()

        # 在球面上生成视角（斐波那契球）
        views = self._fibonacci_sphere(n_views)

        all_descriptors = []
        all_points3d = []

        for vi, (theta, phi) in enumerate(views):
            # 虚拟相机位姿
            R, t = self._camera_at_sphere(theta, phi, distance)

            # 渲染模型
            rendered = self._render_model(R, t, model_scale)
            if rendered is None:
                continue

            # 提取特征
            kp, des = self._detector.detectAndCompute(rendered, None)
            if des is None or len(kp) < 5:
                continue

            # 反向投影：射线与模型求交 → 三维坐标
            points3d, valid_mask = self._back_project(kp, R, t, model_scale)

            if points3d is not None and len(points3d) > 0:
                all_descriptors.append(des[valid_mask])
                all_points3d.append(points3d)

            # 调试：每 20 个视角保存一张渲染图
            if debug and vi % 20 == 0:
                os.makedirs("output", exist_ok=True)
                cv2.imwrite(f"output/model_view_{vi:03d}.png", rendered)

        if not all_descriptors:
            print("特征数据库为空！")
            return 0

        self._db_descriptors = np.vstack(all_descriptors).astype(np.float32)
        self._db_points3d = np.vstack(all_points3d).astype(np.float64)
        self._db_kdtree = KDTree(self._db_points3d)

        elapsed = (time.perf_counter() - t0) * 1000
        print(f"特征数据库: {len(self._db_descriptors)} 个特征点, "
              f"{len(views)} 个视角, {elapsed:.0f} ms")
        return len(self._db_descriptors)

    # ==================================================================
    # 姿态估计
    # ==================================================================

    def estimate_pose(
        self,
        image: np.ndarray,
        min_matches: int = 10,
        reproj_threshold: float = 3.0,
        debug: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        """
        估计模型在单张图像中的姿态。

        Args:
            image: 输入图像 (BGR 或灰度)。
            min_matches: 最少匹配数。
            reproj_threshold: PnP RANSAC 重投影阈值（像素）。
            debug: 保存匹配可视化。

        Returns:
            ``(R, t, n_inliers)``。失败时 R, t 为 None。
        """
        if self._db_descriptors is None:
            print("请先运行 build_database()")
            return None, None, 0

        t0 = time.perf_counter()
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, des = self._detector.detectAndCompute(gray, None)
        if des is None or len(kp) < min_matches:
            return None, None, 0

        # 匹配
        knn = self._matcher.knnMatch(des, self._db_descriptors, k=2)
        good = [m for m, n in knn if m.distance < 0.75 * n.distance]

        if len(good) < min_matches:
            if debug:
                print(f"  匹配不足: {len(good)} < {min_matches}")
            return None, None, len(good)

        # 收集 3D-2D 对应
        obj_pts = []
        img_pts = []
        for m in good:
            obj_pts.append(self._db_points3d[m.trainIdx])
            img_pts.append(kp[m.queryIdx].pt)

        obj_arr = np.array(obj_pts, dtype=np.float64)
        img_arr = np.array(img_pts, dtype=np.float64)

        # PnP + RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_arr, img_arr, self.K, None,
            flags=cv2.SOLVEPNP_EPNP,
            iterationsCount=200,
            reprojectionError=reproj_threshold,
            confidence=0.99,
        )

        n_inl = len(inliers) if inliers is not None else 0

        if not success or n_inl < min_matches:
            if debug:
                print(f"  PnP 失败: {n_inl} 内点")
            return None, None, n_inl

        R, _ = cv2.Rodrigues(rvec)

        elapsed = (time.perf_counter() - t0) * 1000
        if debug:
            print(f"  姿态估计: {n_inl}/{len(good)} 内点, {elapsed:.0f} ms")

        # 可视化
        if debug and R is not None:
            vis = image.copy()
            if vis.ndim == 2:
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            # 画出匹配点
            for m in good[:50]:
                pt = tuple(map(int, kp[m.queryIdx].pt))
                cv2.circle(vis, pt, 2, (0, 255, 0), -1)
            # 画出坐标轴
            cv2.drawFrameAxes(vis, self.K, None, rvec, tvec,
                              self.marker_size * 4, 2)
            cv2.imwrite("output/model_pose_result.png", vis)
            if debug:
                print("  可视化: output/model_pose_result.png")

        return R, tvec.reshape(3,), n_inl

    # ==================================================================
    # 内部方法
    # ==================================================================

    @staticmethod
    def _fibonacci_sphere(n: int) -> List[Tuple[float, float]]:
        """在球面上生成均匀分布的 (theta, phi) 角。"""
        points = []
        phi_golden = math.pi * (3.0 - math.sqrt(5.0))
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y: 1 → -1
            radius = math.sqrt(1 - y * y)
            theta = phi_golden * i
            phi = math.acos(y)
            points.append((theta, phi))
        return points

    def _camera_at_sphere(
        self, theta: float, phi: float, distance: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """在球面坐标处放置相机，看向原点。"""
        cx = distance * math.sin(phi) * math.cos(theta)
        cy = distance * math.sin(phi) * math.sin(theta)
        cz = distance * math.cos(phi)
        cam_pos = np.array([cx, cy, cz])

        z_axis = -cam_pos / np.linalg.norm(cam_pos)
        x_axis = np.cross(np.array([0.0, 1.0, 0.0]), z_axis)
        if np.linalg.norm(x_axis) < 1e-6:
            x_axis = np.cross(np.array([1.0, 0.0, 0.0]), z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        R = np.vstack([x_axis, y_axis, z_axis])
        t = (-R @ cam_pos).reshape(3, 1)
        return R, t

    def _render_model(
        self, R: np.ndarray, t: np.ndarray, scale: float = 1.0,
        image_size: Tuple[int, int] = (640, 480),
    ) -> Optional[np.ndarray]:
        """渲染模型的多尺度线框图像（边缘清晰，适合 SIFT）。"""
        w, h = image_size
        img = np.full((h, w), 255, dtype=np.uint8)

        rvec = cv2.Rodrigues(R)[0]
        tvec = t.reshape(3, 1)
        K_small = self.K.copy()
        K_small[0] *= w / 1280.0
        K_small[1] *= h / 720.0
        K_small = K_small.astype(np.float32)

        # 渲染每个三角面：浅色填充 + 深色边框（线框风格）
        face_depths = []
        for fi, (i0, i1, i2) in enumerate(self.model.faces):
            v0, v1, v2 = [self.model.vertices[i] * scale for i in (i0, i1, i2)]
            center = (v0 + v1 + v2) / 3
            pc = R @ center + tvec.ravel()
            if pc[2] > 0.01:
                face_depths.append((pc[2], fi, (v0, v1, v2)))

        face_depths.sort(key=lambda x: x[0], reverse=True)

        # 第一遍：浅色填充
        for _, fi, (v0, v1, v2) in face_depths:
            verts = np.array([v0, v1, v2], dtype=np.float32).reshape(-1, 1, 3)
            proj, _ = cv2.projectPoints(verts, rvec, tvec, K_small, None)
            pts = proj.reshape(-1, 2).astype(np.int32)
            if np.all(pts > -100) and np.all(pts < [w + 100, h + 100]):
                normal = np.cross(v1 - v0, v2 - v0)
                nn = np.linalg.norm(normal)
                light = (abs(normal[2]) * 0.3 + 0.7) if nn > 1e-9 else 0.7
                gray = int(210 + 40 * light)
                cv2.fillPoly(img, [pts], gray)

        # 第二遍：深色边缘 + 纹理点
        for _, fi, (v0, v1, v2) in face_depths:
            verts = np.array([v0, v1, v2], dtype=np.float32).reshape(-1, 1, 3)
            proj, _ = cv2.projectPoints(verts, rvec, tvec, K_small, None)
            pts = proj.reshape(-1, 2).astype(np.int32)
            if np.all(pts > -100) and np.all(pts < [w + 100, h + 100]):
                # 粗边缘（多尺度线框）
                cv2.polylines(img, [pts], True, 30, 2)
                cv2.polylines(img, [pts], True, 80, 1)
                cv2.polylines(img, [pts], True, 130, 1)

                # 顶点处画高对比度圆点
                for pt in pts:
                    cv2.circle(img, tuple(pt), 4, 0, -1)
                    cv2.circle(img, tuple(pt), 6, 0, 1)

                # 面上撒纹理点
                if fi % 2 == 0:
                    rng = np.random.default_rng(fi * 7)
                    for _ in range(6):
                        r1, r2 = rng.random(), rng.random()
                        if r1 + r2 > 1: r1, r2 = 1-r1, 1-r2
                        pt3d = v0 + r1*(v1-v0) + r2*(v2-v0)
                        pt2d, _ = cv2.projectPoints(
                            np.array([[pt3d]], dtype=np.float32),
                            rvec, tvec, K_small, None)
                        pu, pv = int(pt2d[0,0,0]), int(pt2d[0,0,1])
                        if 0 <= pu < w and 0 <= pv < h:
                            cv2.circle(img, (pu, pv), 3, 0, -1)

        return img

    def _back_project(
        self, keypoints: List[cv2.KeyPoint],
        R: np.ndarray, t: np.ndarray, scale: float = 1.0,
        image_size: Tuple[int, int] = (640, 480),
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        将渲染视图中的关键点反向投影到模型表面，计算三维坐标。

        Returns:
            ``(points3d, valid_mask)``。
            points3d 为 (M, 3)，valid_mask 为布尔数组标记哪些关键点成功投影。
        """
        w, h = image_size
        K_small = self.K.copy()
        K_small[0] *= w / 1280.0
        K_small[1] *= h / 720.0
        K_inv = np.linalg.inv(K_small)

        cam_center = -R.T @ t.ravel()
        points3d = []
        valid = []

        for kp in keypoints:
            # 射线方向（世界坐标系）
            pixel = np.array([kp.pt[0], kp.pt[1], 1.0])
            ray_dir = R.T @ (K_inv @ pixel)
            ray_dir = ray_dir / np.linalg.norm(ray_dir)

            # 与所有面求交，取最近的
            best_t = float("inf")
            best_point = None

            for i0, i1, i2 in self.model.faces:
                v0, v1, v2 = [self.model.vertices[i] * scale for i in (i0, i1, i2)]
                hit, t_val, point = self._ray_triangle(
                    cam_center, ray_dir, v0, v1, v2
                )
                if hit and 0 < t_val < best_t:
                    best_t = t_val
                    best_point = point

            if best_point is not None:
                points3d.append(best_point)
                valid.append(True)
            else:
                valid.append(False)

        if points3d:
            return (np.array(points3d, dtype=np.float64),
                    np.array(valid, dtype=bool))
        return None, None

    @staticmethod
    def _ray_triangle(
        origin: np.ndarray, direction: np.ndarray,
        v0: np.ndarray, v1: np.ndarray, v2: np.ndarray,
    ) -> Tuple[bool, float, Optional[np.ndarray]]:
        """
        Möller-Trumbore 射线-三角形求交。

        Returns:
            ``(hit, t, point)``。
        """
        EPS = 1e-9
        e1 = v1 - v0
        e2 = v2 - v0
        h = np.cross(direction, e2)
        a = np.dot(e1, h)

        if abs(a) < EPS:
            return False, 0, None

        f = 1.0 / a
        s = origin - v0
        u = f * np.dot(s, h)

        if u < 0.0 or u > 1.0:
            return False, 0, None

        q = np.cross(s, e1)
        v = f * np.dot(direction, q)

        if v < 0.0 or u + v > 1.0:
            return False, 0, None

        t = f * np.dot(e2, q)
        if t > EPS:
            point = origin + direction * t
            return True, t, point

        return False, 0, None


# ===================================================================
# 测试
# ===================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

    print("=" * 60)
    print("基于模型的姿态估计 — 测试")
    print("=" * 60)

    # 使用内置飞机模型
    print("\n[1] 加载三维模型 …")
    model = Model3D.from_aircraft_model()

    # 采样表面点
    pts = model.sample_surface_points(200)
    print(f"  表面采样: {len(pts)} 个点")
    print(f"  尺寸: X=[{pts[:,0].min():.3f},{pts[:,0].max():.3f}] "
          f"Y=[{pts[:,1].min():.3f},{pts[:,1].max():.3f}] "
          f"Z=[{pts[:,2].min():.3f},{pts[:,2].max():.3f}]")

    # 构建特征数据库
    print("\n[2] 构建特征数据库 …")
    # 使用合理的相机内参
    K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64)
    tracker = ModelBasedTracker(K, model, marker_size=0.03)

    n_features = tracker.build_database(n_views=80, distance=0.4, debug=True)
    print(f"  数据库特征点: {n_features}")

    if n_features == 0:
        print("特征数据库构建失败 - 渲染视图中未检测到足够特征")
        print("解决方案: 模型表面需要纹理。添加随机纹理或使用较大模型。")
    else:
        print("\n[3] 测试姿态估计 …")
        # 用数据库中的一个虚拟视角来测试
        test_R, test_t = tracker._camera_at_sphere(1.2, 1.5, 0.3)
        test_img = tracker._render_model(test_R, test_t, image_size=(1280, 720))
        if test_img is not None:
            cv2.imwrite("output/model_test_view.png", test_img)
            R_est, t_est, n_inl = tracker.estimate_pose(
                test_img, min_matches=5, debug=True)

            if R_est is not None:
                # 角度误差
                R_err = R_est @ test_R.T
                angle = math.acos(np.clip((np.trace(R_err)-1)/2, -1, 1))
                print(f"  旋转误差: {np.rad2deg(angle):.2f}°")
                print(f"  内点数: {n_inl}")
            else:
                print(f"  姿态估计失败 (匹配数={n_inl})")
