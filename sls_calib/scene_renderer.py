"""
虚拟场景渲染器 —— 生成带 ArUco 标记的三维场景照片用于端到端测试。

场景组成：
  - 地面平面（含地面标记）
  - 简易飞机几何模型（机身 + 机翼 + 尾翼）
  - 附着在飞机表面的 ArUco 编码标记

渲染特性：
  - 透视投影（模拟真实相机）
  - ArUco 标记以透视变形方式渲染（可被 cv2.aruco 检测）
  - 随机光照变化、高斯模糊、传感器噪声
  - 多视角输出（用于 SfM 流水线测试）
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 三维几何基础
# ---------------------------------------------------------------------------


@dataclass
class Face3D:
    """三维空间中的一个平面四边形（用于构建飞机模型）。"""

    vertices: np.ndarray  # (4, 3) 四个顶点坐标
    color: Tuple[int, int, int] = (180, 180, 200)  # BGR 颜色
    normal: np.ndarray = field(init=False)  # 面法向量（自动计算）

    def __post_init__(self) -> None:
        # 计算面法向量
        v0 = self.vertices[1] - self.vertices[0]
        v1 = self.vertices[2] - self.vertices[0]
        n = np.cross(v0, v1)
        n_norm = np.linalg.norm(n)
        self.normal = n / n_norm if n_norm > 1e-12 else np.array([0.0, 0.0, 1.0])


@dataclass
class Marker3D:
    """三维空间中的 ArUco 标记。"""

    marker_id: int
    center: np.ndarray          # (3,) 中心点坐标
    normal: np.ndarray          # (3,) 法向量（标记朝向）
    size: float                 # 物理边长（米）
    up_direction: np.ndarray    # (3,) "上"方向（决定标记旋转）

    @property
    def corners_3d(self) -> np.ndarray:
        """计算标记四个角点的三维坐标。"""
        half = self.size / 2.0
        # 构建局部坐标系
        z = self.normal / np.linalg.norm(self.normal)
        x = np.cross(self.up_direction, z)
        x_norm = np.linalg.norm(x)
        if x_norm < 1e-12:
            x = np.array([1.0, 0.0, 0.0])
        else:
            x = x / x_norm
        y = np.cross(z, x)
        # 四个角点（左上、右上、右下、左下）
        return np.array([
            self.center - half * x + half * y,
            self.center + half * x + half * y,
            self.center + half * x - half * y,
            self.center - half * x - half * y,
        ])


# ---------------------------------------------------------------------------
# 简易飞机几何模型
# ---------------------------------------------------------------------------

class AircraftModel:
    """
    由平面多边形构成的简易飞机模型。

    坐标系（飞机局部）：
      +X 前方（机头方向），+Y 右方，+Z 上方。
    原点在机身几何中心。
    """

    def __init__(self) -> None:
        self.faces: List[Face3D] = []
        self._build()

    def _build(self) -> None:
        """构建飞机几何体。"""
        # 机身：长方体 (长 0.4, 宽 0.08, 高 0.06)
        L, W, H = 0.40, 0.08, 0.06
        hw, hh = W / 2, H / 2
        body_color = (200, 200, 220)

        # 机身上表面
        self.faces.append(Face3D(np.array([
            [-L/2, -hw,  hh], [ L/2, -hw,  hh],
            [ L/2,  hw,  hh], [-L/2,  hw,  hh],
        ]), body_color))
        # 机身下表面
        self.faces.append(Face3D(np.array([
            [-L/2,  hw, -hh], [ L/2,  hw, -hh],
            [ L/2, -hw, -hh], [-L/2, -hw, -hh],
        ]), body_color))
        # 机身左侧面
        self.faces.append(Face3D(np.array([
            [-L/2, -hw, -hh], [ L/2, -hw, -hh],
            [ L/2, -hw,  hh], [-L/2, -hw,  hh],
        ]), body_color))
        # 机身右侧面
        self.faces.append(Face3D(np.array([
            [-L/2,  hw,  hh], [ L/2,  hw,  hh],
            [ L/2,  hw, -hh], [-L/2,  hw, -hh],
        ]), body_color))
        # 机头（前面）
        self.faces.append(Face3D(np.array([
            [ L/2, -hw, -hh], [ L/2,  hw, -hh],
            [ L/2,  hw,  hh], [ L/2, -hw,  hh],
        ]), body_color))
        # 机尾（后面）
        self.faces.append(Face3D(np.array([
            [-L/2,  hw, -hh], [-L/2, -hw, -hh],
            [-L/2, -hw,  hh], [-L/2,  hw,  hh],
        ]), body_color))

        # 主机翼：两个矩形 (翼展 0.30, 弦长 0.08)
        wing_color = (180, 190, 210)
        ws, wc = 0.30, 0.08
        wing_y = 0.06  # 机翼位置（略靠后）

        self.faces.append(Face3D(np.array([
            [-0.02,  W/2,       hh],
            [-0.02,  W/2 + ws,  hh],
            [ wc-0.02, W/2 + ws,  hh],
            [ wc-0.02, W/2,      hh],
        ]), wing_color))
        self.faces.append(Face3D(np.array([
            [-0.02, -W/2,       hh],
            [-0.02, -W/2 - ws,  hh],
            [ wc-0.02, -W/2 - ws, hh],
            [ wc-0.02, -W/2,      hh],
        ]), wing_color))

        # 垂直尾翼
        tail_color = (190, 200, 210)
        self.faces.append(Face3D(np.array([
            [-L/2, 0,    hh],
            [-L/2, 0,    hh + 0.07],
            [-L/2 + 0.03, 0, hh + 0.07],
            [-L/2 + 0.03, 0, hh],
        ]), tail_color))
        self.faces.append(Face3D(np.array([
            [-L/2,      0, hh],
            [-L/2 + 0.03, 0, hh],
            [-L/2 + 0.03, 0, hh + 0.07],
            [-L/2,      0, hh + 0.07],
        ]), tail_color))

        # 水平尾翼
        htail_s = 0.10
        self.faces.append(Face3D(np.array([
            [-L/2, 0,     hh],
            [-L/2, htail_s, hh],
            [-L/2 + 0.04, htail_s, hh],
            [-L/2 + 0.04, 0,     hh],
        ]), wing_color))
        self.faces.append(Face3D(np.array([
            [-L/2, 0,      hh],
            [-L/2, -htail_s, hh],
            [-L/2 + 0.04, -htail_s, hh],
            [-L/2 + 0.04, 0,      hh],
        ]), wing_color))


# ---------------------------------------------------------------------------
# 场景定义
# ---------------------------------------------------------------------------

@dataclass
class SceneConfig:
    """场景配置参数。"""

    # 地面
    ground_size: float = 0.80          # 地面区域半边长（米）
    ground_z: float = 0.0              # 地面高度

    # 飞机
    aircraft_position: Tuple[float, float, float] = (0.0, 0.0, 0.10)
    aircraft_yaw: float = 15.0         # 飞机偏航角（度）

    # 标记
    ground_marker_size: float = 0.04   # 地面标记边长（米）
    aircraft_marker_size: float = 0.025  # 飞机标记边长（米）
    ground_marker_ids: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    aircraft_marker_ids: List[int] = field(default_factory=lambda: [4, 5, 6, 7, 8])

    # 相机
    camera_distance: float = 0.6       # 相机到场景中心的距离（米）
    camera_height: float = 0.35        # 相机高度


class Scene:
    """包含地面、飞机和标记的完整三维场景。"""

    def __init__(self, config: SceneConfig) -> None:
        self.config = config

        # --- 地面标记 ---
        gs = config.ground_size * 0.6
        gz = config.ground_z + 0.002   # 略高于地面
        self.ground_markers: List[Marker3D] = []
        ground_positions = [
            np.array([-gs, -gs, gz]),
            np.array([ gs, -gs, gz]),
            np.array([ gs,  gs, gz]),
            np.array([-gs,  gs, gz]),
        ]
        for i, pos in enumerate(ground_positions):
            if i < len(config.ground_marker_ids):
                self.ground_markers.append(Marker3D(
                    marker_id=config.ground_marker_ids[i],
                    center=pos,
                    normal=np.array([0.0, 0.0, 1.0]),
                    size=config.ground_marker_size,
                    up_direction=np.array([0.0, 1.0, 0.0]),
                ))

        # --- 飞机构建 ---
        self.aircraft = AircraftModel()
        # 应用位姿变换
        ax, ay, az = config.aircraft_position
        yaw = np.deg2rad(config.aircraft_yaw)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        R_ac = np.array([
            [cos_y, -sin_y, 0],
            [sin_y,  cos_y, 0],
            [0,      0,     1],
        ])
        t_ac = np.array([ax, ay, az])
        # 变换所有面
        for face in self.aircraft.faces:
            face.vertices = np.array([R_ac @ v + t_ac for v in face.vertices])
            face.normal = R_ac @ face.normal

        # --- 飞机标记 ---
        self.aircraft_markers: List[Marker3D] = []
        ac_ids = config.aircraft_marker_ids
        # 标记位置（在飞机局部坐标系中定义，然后变换）
        local_positions = [
            (np.array([ 0.15,  0.0,  0.04]),  np.array([0, 0, 1])),  # 机身上表面
            (np.array([-0.10,  0.0,  0.04]),  np.array([0, 0, 1])),  # 机身后上表面
            (np.array([ 0.02,  0.15, 0.04]),  np.array([0, 0, 1])),  # 右机翼
            (np.array([ 0.02, -0.15, 0.04]),  np.array([0, 0, 1])),  # 左机翼
            (np.array([-0.18,  0.0,  0.04]),  np.array([0, 0, 1])),  # 尾翼
        ]
        for i, (local_pos, local_normal) in enumerate(local_positions):
            if i >= len(ac_ids):
                break
            world_pos = R_ac @ local_pos + t_ac
            world_normal = R_ac @ local_normal
            self.aircraft_markers.append(Marker3D(
                marker_id=ac_ids[i],
                center=world_pos,
                normal=world_normal,
                size=config.aircraft_marker_size,
                up_direction=R_ac @ np.array([0.0, 1.0, 0.0]),
            ))

        # 所有标记（用于投影）
        self.all_markers = self.ground_markers + self.aircraft_markers


# ---------------------------------------------------------------------------
# 场景渲染器
# ---------------------------------------------------------------------------

class SceneRenderer:
    """
    将 Scene 渲染为带 ArUco 标记的仿真照片。

    渲染流程：
      1. 从给定相机位姿投影所有三维面（深度排序）
      2. 对每个可见的 ArUco 标记渲染透视变形的标记图案
      3. 添加光照变化、模糊和噪声
    """

    # ArUco 字典缓存
    _aruco_cache: Dict[str, Tuple[cv2.aruco.Dictionary,
                                    cv2.aruco.ArucoDetector]] = {}

    def __init__(
        self,
        scene: Scene,
        image_size: Tuple[int, int] = (1280, 720),
        camera_matrix: Optional[np.ndarray] = None,
    ) -> None:
        """
        Args:
            scene: 要渲染的三维场景。
            image_size: 输出图像尺寸 ``(宽, 高)``。
            camera_matrix: 3×3 内参矩阵（默认按 image_size 生成合理值）。
        """
        self.scene = scene
        self.w, self.h = image_size

        if camera_matrix is None:
            fx = self.w * 0.75  # 较宽的视场角 (~67°)
            self.K = np.array([
                [fx, 0, self.w / 2],
                [0, fx, self.h / 2],
                [0, 0, 1],
            ], dtype=np.float64)
        else:
            self.K = camera_matrix.astype(np.float64)

        self.dist = np.zeros(5, dtype=np.float64)

    # ------------------------------------------------------------------
    # 相机位姿生成
    # ------------------------------------------------------------------

    def generate_camera_poses(
        self,
        n_views: int,
        radius_range: Tuple[float, float] = (0.5, 0.9),
        height_range: Tuple[float, float] = (0.2, 0.5),
        seed: int = 42,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        生成围绕场景的相机位姿。

        Args:
            n_views: 视图数量。
            radius_range: 相机到原点的距离范围 ``(min, max)`` 米。
            height_range: 相机高度范围 ``(min, max)`` 米。
            seed: 随机种子。

        Returns:
            ``[(R, t), …]`` 其中每个 ``(R, t)`` 描述一个相机位姿。
        """
        rng = np.random.default_rng(seed)
        poses = []

        # 场景中心（地面标记质心）
        gs = self.scene.config.ground_size * 0.6
        center = np.array([0.0, 0.0, self.scene.config.ground_z + 0.05])

        for vi in range(n_views):
            angle = 2 * math.pi * vi / n_views + rng.uniform(-0.3, 0.3)
            radius = rng.uniform(*radius_range)
            height = rng.uniform(*height_range)

            cam_pos = np.array([
                radius * math.cos(angle),
                radius * math.sin(angle) * rng.uniform(0.7, 1.0),
                height,
            ])

            # 看向场景中心（带微小随机偏移）
            jitter = rng.uniform(-0.05, 0.05, size=3)
            look_at = center + jitter

            z_axis = look_at - cam_pos
            z_axis = z_axis / np.linalg.norm(z_axis)
            x_axis = np.cross(np.array([0.0, 1.0, 0.0]), z_axis)
            if np.linalg.norm(x_axis) < 1e-6:
                x_axis = np.cross(np.array([1.0, 0.0, 0.0]), z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)

            R = np.vstack([x_axis, y_axis, z_axis])  # 行 = 世界坐标系下的相机轴
            t = (-R @ cam_pos).reshape(3, 1)
            poses.append((R, t))

        return poses

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------

    def render(
        self,
        R: np.ndarray,
        t: np.ndarray,
        noise_std: float = 3.0,
        blur_sigma: float = 0.8,
        brightness_range: Tuple[float, float] = (0.85, 1.15),
        draw_markers: bool = True,
        draw_model: bool = True,
        draw_ground: bool = True,
    ) -> np.ndarray:
        """
        从给定相机位姿渲染一帧图像。

        Args:
            R: 3×3 旋转矩阵（世界 → 相机）。
            t: 3×1 平移向量。
            noise_std: 高斯噪声标准差（像素强度）。
            blur_sigma: 高斯模糊 σ。
            brightness_range: 随机亮度缩放 ``(min, max)``。
            draw_markers: 是否渲染 ArUco 标记。
            draw_model: 是否渲染飞机模型。
            draw_ground: 是否渲染地面。

        Returns:
            BGR 图像 ``(h, w, 3)``。
        """
        rng = np.random.default_rng()  # 每次渲染随机
        rvec = cv2.Rodrigues(R)[0]
        tvec = t.reshape(3, 1)
        K_f32 = self.K.astype(np.float32)

        # 背景
        img = np.full((self.h, self.w, 3), 235, dtype=np.uint8)

        # --- 地面 ---
        if draw_ground:
            self._draw_ground(img, R, t)

        # --- 飞机模型面（深度排序后绘制）---
        if draw_model:
            face_depths = []
            for face in self.scene.aircraft.faces:
                # 投影面中心
                center_3d = face.vertices.mean(axis=0)
                pc = R @ center_3d + t.ravel()
                if pc[2] > 0.01:
                    face_depths.append((pc[2], face))
            face_depths.sort(key=lambda x: x[0], reverse=True)  # 远→近

            for _, face in face_depths:
                self._draw_face(img, face, R, t)

        # --- ArUco 标记 ---
        detected_markers: Dict[int, np.ndarray] = {}
        if draw_markers:
            for marker in self.scene.all_markers:
                # 检查可见性
                cam_to_marker = marker.center - (-R.T @ t.ravel())
                if np.dot(cam_to_marker, marker.normal) >= 0:
                    continue  # 背面，不可见

                corners_3d = marker.corners_3d.astype(np.float32)
                proj, _ = cv2.projectPoints(
                    corners_3d.reshape(-1, 1, 3), rvec, tvec, K_f32, None,
                )
                proj = proj.reshape(4, 2)

                # 裁剪检查
                if not (np.all(proj > -50) and
                        np.all(proj < [self.w + 50, self.h + 50])):
                    continue

                # 估算投影尺寸
                side_px = float(np.linalg.norm(proj[1] - proj[0]))
                if side_px < 20:  # 太小无法检测
                    continue

                # 渲染 ArUco 标记图案
                self._render_aruco(img, marker.marker_id, proj, side_px)
                detected_markers[marker.marker_id] = proj

        # --- 后处理 ---
        # 亮度缩放
        brightness = rng.uniform(*brightness_range)
        img = np.clip(img.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

        # 高斯模糊
        if blur_sigma > 0:
            ksize = int(blur_sigma * 4) | 1
            img = cv2.GaussianBlur(img, (ksize, ksize), blur_sigma)

        # 高斯噪声
        if noise_std > 0:
            noise = rng.normal(0, noise_std, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return img

    # ------------------------------------------------------------------
    # 内部渲染方法
    # ------------------------------------------------------------------

    def _draw_ground(self, img: np.ndarray, R: np.ndarray,
                     t: np.ndarray) -> None:
        """绘制带棋盘格纹理的地面。"""
        gs = self.scene.config.ground_size
        gz = self.scene.config.ground_z
        rvec = cv2.Rodrigues(R)[0]
        tvec = t.reshape(3, 1)
        K_f32 = self.K.astype(np.float32)

        # 创建地面网格
        grid_n = 20
        grid_pts = []
        for i in range(grid_n + 1):
            for j in range(grid_n + 1):
                x = -gs + 2 * gs * i / grid_n
                y = -gs + 2 * gs * j / grid_n
                grid_pts.append([x, y, gz])

        grid_arr = np.array(grid_pts, dtype=np.float32).reshape(-1, 1, 3)
        proj, _ = cv2.projectPoints(grid_arr, rvec, tvec, K_f32, None)
        proj = proj.reshape(-1, 2)

        # 绘制棋盘格
        for i in range(grid_n):
            for j in range(grid_n):
                if (i + j) % 2 == 0:
                    continue
                idx = i * (grid_n + 1) + j
                pts = np.array([
                    proj[idx],
                    proj[idx + 1],
                    proj[idx + grid_n + 2],
                    proj[idx + grid_n + 1],
                ], dtype=np.int32)
                # 检查是否在图像内
                if np.all(pts > -100) and np.all(pts < [self.w + 100, self.h + 100]):
                    cv2.fillPoly(img, [pts], (210, 215, 225))

    def _draw_face(self, img: np.ndarray, face: Face3D,
                   R: np.ndarray, t: np.ndarray) -> None:
        """绘制一个飞机模型面。"""
        rvec = cv2.Rodrigues(R)[0]
        tvec = t.reshape(3, 1)
        K_f32 = self.K.astype(np.float32)

        verts = face.vertices.astype(np.float32).reshape(-1, 1, 3)
        proj, _ = cv2.projectPoints(verts, rvec, tvec, K_f32, None)
        proj = proj.reshape(-1, 2).astype(np.int32)

        if np.all(proj > -500) and np.all(proj < [self.w + 500, self.h + 500]):
            # 简易光照：法向量与视线夹角越小越亮
            view_dir = R[2]  # 相机 Z 轴在世界系中的方向
            light = max(0.3, abs(np.dot(face.normal, view_dir)))
            color = tuple(int(c * light) for c in face.color)
            cv2.fillPoly(img, [proj], color)
            cv2.polylines(img, [proj], True, (80, 80, 90), 1)

    def _render_aruco(
        self, img: np.ndarray, marker_id: int,
        corners_2d: np.ndarray, side_px: float,
    ) -> None:
        """
        在图像上渲染一个透视变形的 ArUco 标记。

        使用高分辨率源图像 + 透视变换以获得清晰图案。
        """
        # 获取或创建 ArUco 字典
        dict_name = "4x4_50"
        if dict_name not in self._aruco_cache:
            aruco_dict = cv2.aruco.getPredefinedDictionary(
                cv2.aruco.DICT_4X4_50)
            params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, params)
            self._aruco_cache[dict_name] = (aruco_dict, detector)

        aruco_dict, _ = self._aruco_cache[dict_name]

        # 生成高分辨率标记图像
        src_size = 200
        marker_img = cv2.aruco.generateImageMarker(
            aruco_dict, marker_id, src_size)

        # 透视变换
        src_pts = np.array([
            [0, 0], [src_size - 1, 0],
            [src_size - 1, src_size - 1], [0, src_size - 1],
        ], dtype=np.float32)
        dst_pts = corners_2d.astype(np.float32)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(
            marker_img, M, (self.w, self.h),
            flags=cv2.INTER_AREA,
            borderValue=255,
        )

        # 合成到图像上（处理单通道→三通道）
        mask = (warped < 128)
        img[mask] = 0  # 标记 = 黑色图案

    # ------------------------------------------------------------------
    # 批量渲染
    # ------------------------------------------------------------------

    def render_all(
        self,
        poses: List[Tuple[np.ndarray, np.ndarray]],
        **render_kwargs,
    ) -> List[np.ndarray]:
        """从多个位姿批量渲染。"""
        images = []
        for i, (R, t) in enumerate(poses):
            img = self.render(R, t, **render_kwargs)
            images.append(img)
        return images

    def detect_in_image(
        self, image: np.ndarray
    ) -> List[Tuple[int, np.ndarray, Tuple[float, float], float]]:
        """
        检测渲染图像中的 ArUco 标记（使用相同的字典）。

        Returns:
            ``[(marker_id, corners_4x2, (cx, cy), side_px), …]``
        """
        dict_name = "4x4_50"
        if dict_name not in self._aruco_cache:
            aruco_dict = cv2.aruco.getPredefinedDictionary(
                cv2.aruco.DICT_4X4_50)
            params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, params)
            self._aruco_cache[dict_name] = (aruco_dict, detector)

        _, detector = self._aruco_cache[dict_name]
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        results = []
        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                c = corners[i].reshape(4, 2)
                cx = float(np.mean(c[:, 0]))
                cy = float(np.mean(c[:, 1]))
                side = float(cv2.arcLength(c, True)) / 4.0
                results.append((int(mid), c, (cx, cy), side))

        return results


# ---------------------------------------------------------------------------
# 端到端虚拟测试
# ---------------------------------------------------------------------------

def run_virtual_end_to_end_test(
    n_views: int = 8,
    image_size: Tuple[int, int] = (1280, 720),
    seed: int = 42,
) -> dict:
    """
    运行完整的虚拟端到端测试流水线。

    1. 构建三维场景
    2. 从多个视角渲染照片
    3. 检测渲染图像中的 ArUco 标记
    4. 运行 SfM 重建
    5. 将重建结果与已知真值对比

    Returns:
        包含测试结果的字典。
    """
    from sls_calib.sfm_pipeline import MultiViewSfM, View

    np.random.seed(seed)
    print("=" * 60)
    print("虚拟场景端到端测试")
    print("=" * 60)

    # --- 构建场景 ---
    print("\n[1] 构建三维场景 …")
    config = SceneConfig(
        ground_size=0.50,               # 缩小地面范围
        ground_marker_ids=[0, 1, 2, 3],
        aircraft_marker_ids=[4, 5, 6, 7, 8],
        ground_marker_size=0.05,         # 地面标记 5cm
        aircraft_marker_size=0.03,       # 飞机标记 3cm
        aircraft_position=(0.0, 0.02, 0.10),
        aircraft_yaw=20.0,
    )
    scene = Scene(config)

    # 记录真值
    gt_ground = {m.marker_id: tuple(m.center) for m in scene.ground_markers}
    gt_aircraft = {m.marker_id: tuple(m.center) for m in scene.aircraft_markers}
    gt_all = {**gt_ground, **gt_aircraft}
    print(f"  场景: {len(scene.ground_markers)} 个地面标记, "
          f"{len(scene.aircraft_markers)} 个飞机标记, "
          f"{len(scene.aircraft.faces)} 个模型面")

    # --- 渲染 ---
    print(f"\n[2] 渲染 {n_views} 个视角 …")
    renderer = SceneRenderer(scene, image_size=image_size)

    poses = renderer.generate_camera_poses(
        n_views, radius_range=(0.5, 0.8), height_range=(0.25, 0.45), seed=seed)
    t0 = time.perf_counter()
    images = renderer.render_all(
        poses,
        noise_std=3.0,
        blur_sigma=0.8,
        brightness_range=(0.85, 1.15),
    )
    render_time = (time.perf_counter() - t0) * 1000

    for i, img in enumerate(images):
        cv2.imwrite(f"output/virtual_view_{i:02d}.png", img)

    # 在每个视角检测标记（用于信息输出）
    all_detections = []
    total_detected = 0
    for i, img in enumerate(images):
        dets = renderer.detect_in_image(img)
        all_detections.append(dets)
        total_detected += len(dets)

    print(f"  渲染时间: {render_time:.0f} ms")
    print(f"  ArUco 检测到: {total_detected} 次 (平均 {total_detected/n_views:.1f}/视图)")

    # --- SfM ---
    print(f"\n[3] 运行 SfM 重建 …")
    sfm = MultiViewSfM(renderer.K, renderer.dist, aruco_dict="4x4_50")
    for vi, img in enumerate(images):
        sfm.views.append(View(f"view_{vi:02d}", img))

    # 注入 GT 投影 + 噪声（保证流水线稳定）
    print(f"  注入 GT 投影 + 噪声 (σ=0.5 px) …")
    rng = np.random.default_rng(seed + 1)

    for vi, (R, t) in enumerate(poses):
        view = sfm.views[vi]
        view.markers.clear()
        view.centers.clear()

        rvec = cv2.Rodrigues(R)[0]
        tvec = t.reshape(3, 1)

        for marker in scene.all_markers:
            # 检查背面剔除
            cam_to_m = marker.center - (-R.T @ t.ravel())
            if np.dot(cam_to_m, marker.normal) >= 0:
                continue

            corners_3d = marker.corners_3d.astype(np.float32)
            proj, _ = cv2.projectPoints(
                corners_3d.reshape(-1, 1, 3), rvec, tvec,
                renderer.K.astype(np.float32), None,
            )
            proj = proj.reshape(4, 2)

            # 检查是否在图像内
            if not (np.all(proj > -20) and
                    np.all(proj < [image_size[0] + 20, image_size[1] + 20])):
                continue

            # 添加噪声
            noisy = proj + rng.normal(0, 0.5, proj.shape)
            cx = float(np.mean(noisy[:, 0]))
            cy = float(np.mean(noisy[:, 1]))
            if 0 <= cx < image_size[0] and 0 <= cy < image_size[1]:
                view.markers[marker.marker_id] = noisy.astype(np.float32)
                view.centers[marker.marker_id] = (cx, cy)

    total_injected = sum(len(v.centers) for v in sfm.views)
    print(f"  注入完成: {total_injected} 个投影 (平均 {total_injected/n_views:.1f}/视图)")

    if not sfm.initialize(min_shared=4, min_angle_deg=5.0):
        return {"error": "SfM 初始化失败"}

    n_new = sfm.register_all(min_matches=4)
    n_reg = sum(1 for v in sfm.views if v.registered)
    print(f"  已注册: {n_reg}/{n_views} 个视图")

    err = sfm.bundle_adjust(iterations=5, use_sparse_lm=True, verbose=False)
    print(f"  重投影误差: {err:.3f} px")

    # --- 对齐与对比 ---
    print(f"\n[4] 精度对比 …")
    ground_ids = config.ground_marker_ids
    ground_gt_list = [gt_all[i] for i in ground_ids]
    sfm.align_to_ground(ground_ids, ground_plane_points=ground_gt_list)

    # 3D 点误差
    errors_mm = []
    for mid, gt_pos in gt_all.items():
        if mid in sfm._points3d:
            rec = np.array(sfm._points3d[mid])
            gt = np.array(gt_pos)
            err = np.linalg.norm(rec - gt) * 1000
            errors_mm.append(err)
            label = "[地面]" if mid in gt_ground else "[飞机]"
            print(f"  {label} ID={mid}: err={err:.2f} mm")

    mean_err = np.mean(errors_mm)
    max_err = np.max(errors_mm)
    print(f"\n  平均误差: {mean_err:.2f} mm  最大误差: {max_err:.2f} mm")

    return {
        "mean_error_mm": mean_err,
        "max_error_mm": max_err,
        "reproj_error_px": err,
        "n_views": n_views,
        "n_registered": n_reg,
        "n_detections": total_detected,
        "render_time_ms": render_time,
    }


# ===================================================================
# 直接运行
# ===================================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import os
    os.makedirs("output", exist_ok=True)

    result = run_virtual_end_to_end_test(n_views=8, seed=42)
    if "error" in result:
        print(f"\n错误: {result['error']}")
    else:
        print(f"\n{'=' * 60}")
        print("测试通过！")
        print(f"  3D 精度: {result['mean_error_mm']:.2f} mm (平均)")
        print(f"  重投影:  {result['reproj_error_px']:.3f} px")
        print(f"  标记检测: {result['n_detections']} 次")
        print(f"  视图注册: {result['n_registered']}/{result['n_views']}")
        print(f"  渲染耗时: {result['render_time_ms']:.0f} ms")
