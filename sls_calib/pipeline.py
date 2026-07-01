"""
端到端标定流水线。

统筹完整工作流程：
  相机标定 → 立体标定 → SfM → 姿态估计。
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .marker_detector import SLSMarkerDetector
from .camera_calib import CalibImage, Calibrator
from .coded_marker import CodedMarkerDetector
from .sfm_pipeline import MultiViewSfM, View


class CalibrationPipeline:
    """
    SLS 标定与姿态估计工作流程的一站式入口。

    用法::

        pipeline = CalibrationPipeline({
            "data_dir": "data/my_scene",
            "output_dir": "output/my_scene",
            "marker_size_m": 0.025,
            "aruco_dict": "4x4_50",
            "ground_marker_ids": [0, 1, 2, 3],
            "aircraft_marker_ids": [4, 5, 6],
        })
        results = pipeline.run_all()
        pipeline.export_results()

    Parameters
    ----------
    config : dict
    """

    def __init__(self, config: dict) -> None:
        self.config = config

        # 路径
        self.data_dir = Path(config.get("data_dir", "data"))
        self.output_dir = Path(config.get("output_dir", "output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 标志点配置
        self.marker_size_m = config.get("marker_size_m")
        self.aruco_dict = config.get("aruco_dict", "4x4_50")
        self.ground_ids = config.get("ground_marker_ids", [])
        self.aircraft_ids = config.get("aircraft_marker_ids", [])

        # 相机内参（由 calibrate_camera / load 填充）
        self._K: Optional[np.ndarray] = None
        self._dist: Optional[np.ndarray] = None
        self._image_size: Optional[Tuple[int, int]] = None

        # 流水线状态
        self._sfm: Optional[MultiViewSfM] = None
        self._aircraft_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._timings: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # 步骤 1 —— 相机内参标定
    # ------------------------------------------------------------------

    def calibrate_camera(
        self,
        calib_images: Optional[List[np.ndarray]] = None,
        circle_interval: float = 35.0,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        使用 SLS 圆点网格图像标定单台相机。

        Args:
            calib_images: 标定板图像列表。
            circle_interval: 物理圆点间距（mm）。

        Returns:
            ``(K, dist_coeffs)``。
        """
        if calib_images is None:
            calib_images = self._load_images(
                self.data_dir / "calibration", "calib_*.png"
            )
        if not calib_images:
            print("未找到标定图像。")
            return None, None

        t0 = time.perf_counter()
        calib_imgs = []
        for i, img in enumerate(calib_images):
            ci = CalibImage(name=f"calib_{i}", image=img, selected=True)
            calib_imgs.append(ci)

        calib = Calibrator()
        err = calib.extract_circles(calib_imgs, only_selected=True,
                                     smooth=True, debug=False)
        if err:
            print(f"圆点检测错误：{err}")
            return None, None

        for ci in calib_imgs:
            err = ci.find_circle_indices(circle_interval, debug=False)
            if err:
                print(f"{ci.name} 的网格分配错误：{err}")

        report, K, dist = calib.calibrate_camera(calib_imgs, "calib", debug=True)
        print(report)

        if K is not None:
            self._K = K
            self._dist = dist
            if calib_images:
                h, w = calib_images[0].shape[:2]
                self._image_size = (w, h)

        self._timings["calibrate_camera"] = (time.perf_counter() - t0) * 1000
        return K, dist

    # ------------------------------------------------------------------
    # 步骤 2 —— 加载已有标定结果
    # ------------------------------------------------------------------

    def load_calibration(self, npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        从 ``.npz`` 文件中加载相机内参。

        文件应包含 ``camera_matrix`` 和 ``dist_coeffs``
        数组（与 ``run_calibration.py`` 或
        ``cv2.calibrateCamera`` 保存的格式相同）。
        """
        data = np.load(npz_path)
        self._K = data["camera_matrix"]
        self._dist = data["dist_coeffs"].ravel()
        return self._K, self._dist

    # ------------------------------------------------------------------
    # 步骤 3 —— 多视图 SfM（Phase A）
    # ------------------------------------------------------------------

    def run_sfm(
        self,
        images: Optional[List[np.ndarray]] = None,
    ) -> Optional[MultiViewSfM]:
        """
        运行多视图 SfM 重建。

        需要先设置相机内参（通过 ``calibrate_camera``
        或 ``load_calibration``）。

        Args:
            images: 来自相机 1 的场景图像列表（多视图）。

        Returns:
            ``MultiViewSfM``，包含重建的三维点和相机姿态。
        """
        if self._K is None:
            print("相机内参未设置。请先运行 calibrate_camera()。")
            return None

        if images is None:
            images = self._load_images(self.data_dir / "sfm", "view_*.png")
        if len(images) < 2:
            print("SfM 至少需要 2 张图像。")
            return None

        t0 = time.perf_counter()

        sfm = MultiViewSfM(
            self._K, self._dist,
            marker_size_m=self.marker_size_m,
            aruco_dict=self.aruco_dict,
        )
        sfm.add_views(images)

        if not sfm.initialize():
            print("SfM 初始化失败。")
            return None

        n_new = sfm.register_all()
        print(f"已注册 {n_new} 个额外视图 "
              f"({sum(1 for v in sfm.views if v.registered)}/"
              f"{len(sfm.views)} 个总计)")

        err = sfm.bundle_adjust(iterations=5, use_sparse_lm=True, verbose=True)
        print(f"BA 重投影误差：{err:.3f} px")

        # 对齐到地面
        if self.ground_ids:
            sfm.align_to_ground(self.ground_ids)
            print(f"已与 {len(self.ground_ids)} 个地面标志点对齐")

        self._sfm = sfm
        self._timings["run_sfm"] = (time.perf_counter() - t0) * 1000
        print(sfm.summary())
        return sfm

    # ------------------------------------------------------------------
    # 步骤 4 —— 飞行器姿态估计
    # ------------------------------------------------------------------

    def estimate_aircraft_pose(
        self,
        image: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        通过 PnP 结合已知三维点估计飞行器姿态（Phase B）。

        需要已运行 SfM（``run_sfm``）。

        Args:
            image: 来自相机 2 的单张图像（任意角度）。

        Returns:
            ``(R_aircraft_in_world, t_aircraft_in_world)``。
        """
        if self._sfm is None:
            print("尚未运行 SfM。请先调用 run_sfm()。")
            return None
        if not self.aircraft_ids:
            print("未配置 aircraft_marker_ids。")
            return None

        if image is None:
            images = self._load_images(self.data_dir / "inference", "cam2_*.png")
            if not images:
                print("未找到推断图像。")
                return None
            image = images[0]

        t0 = time.perf_counter()

        # 对于 `get_aircraft_pose_pnp`，需要飞行器标志点的局部坐标。
        # 使用重建的三维坐标作为"局部"坐标（在没有 CAD 模型时的替代方案）。
        local_coords = [
            self._sfm.points_3d.get(mid, (0.0, 0.0, 0.0))
            for mid in self.aircraft_ids
        ]

        R_ac, t_ac = self._sfm.get_aircraft_pose_pnp(
            self.aircraft_ids, local_coords, image,
        )

        if R_ac is None:
            print("飞行器姿态估计失败。")
            return None

        self._aircraft_pose = (R_ac, t_ac)
        self._timings["estimate_aircraft_pose"] = (
            time.perf_counter() - t0
        ) * 1000

        # 欧拉角，便于阅读
        sy = np.sqrt(R_ac[0, 0] ** 2 + R_ac[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            rx = math.degrees(math.atan2(R_ac[2, 1], R_ac[2, 2]))
            ry = math.degrees(math.atan2(-R_ac[2, 0], sy))
            rz = math.degrees(math.atan2(R_ac[1, 0], R_ac[0, 0]))
        else:
            rx = math.degrees(math.atan2(-R_ac[1, 2], R_ac[1, 1]))
            ry = math.degrees(math.atan2(-R_ac[2, 0], sy))
            rz = 0.0

        print(f"飞行器姿态：")
        print(f"  t (m):     ({t_ac[0]:.4f}, {t_ac[1]:.4f}, {t_ac[2]:.4f})")
        print(f"  欧拉角 (°): roll={rx:.2f}  pitch={ry:.2f}  yaw={rz:.2f}")

        return R_ac, t_ac

    # ------------------------------------------------------------------
    # 一键运行全部
    # ------------------------------------------------------------------

    def run_all(
        self,
        calib_images: Optional[List[np.ndarray]] = None,
        sfm_images: Optional[List[np.ndarray]] = None,
        inference_image: Optional[np.ndarray] = None,
        calib_npz: Optional[str] = None,
    ) -> dict:
        """
        运行完整流水线：标定 → SfM → 姿态估计。

        Returns:
            字典，键包括：``"K"``, ``"dist"``, ``"points_3d"``,
            ``"camera_poses"``, ``"aircraft_pose"``, ``"timings"``。
        """
        print("=" * 60)
        print("SLS 标定流水线")
        print("=" * 60)

        # 相机标定
        print("\n--- 步骤 1：相机标定 ---")
        if calib_npz:
            self.load_calibration(calib_npz)
            print(f"已从 {calib_npz} 加载内参")
        else:
            K, dist = self.calibrate_camera(calib_images)
            if K is None:
                return {"error": "相机标定失败"}

        # SfM
        print("\n--- 步骤 2：多视图 SfM ---")
        sfm = self.run_sfm(sfm_images)
        if sfm is None:
            return {"error": "SfM 重建失败"}

        # 飞行器姿态
        print("\n--- 步骤 3：飞行器姿态估计 ---")
        pose = self.estimate_aircraft_pose(inference_image)
        if pose is None:
            return {"error": "飞行器姿态估计失败"}

        print("\n" + "=" * 60)
        print("流水线运行完毕。")
        for step, ms in self._timings.items():
            print(f"  {step}: {ms:.0f} ms")
        print("=" * 60)

        return {
            "K": self._K,
            "dist": self._dist,
            "points_3d": self._sfm.points_3d if self._sfm else {},
            "camera_poses": self._sfm.camera_poses if self._sfm else [],
            "aircraft_pose": pose,
            "timings": self._timings,
        }

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------

    def export_results(self) -> None:
        """将所有流水线结果保存到 ``output_dir``。"""
        if self._K is not None:
            np.savez(
                self.output_dir / "camera.npz",
                camera_matrix=self._K,
                dist_coeffs=self._dist,
            )

        if self._sfm is not None:
            pts = self._sfm.points_3d
            # 将字典转换为数组以便保存
            ids = sorted(pts.keys())
            arr = np.array([pts[i] for i in ids], dtype=np.float64)
            np.savez(
                self.output_dir / "sfm_points.npz",
                marker_ids=np.array(ids),
                points_3d=arr,
            )

        if self._aircraft_pose is not None:
            R, t = self._aircraft_pose
            np.savez(
                self.output_dir / "aircraft_pose.npz",
                R=R, t=t.ravel(),
            )

        # 摘要 JSON
        summary = {
            "n_views": len(self._sfm.views) if self._sfm else 0,
            "n_registered": sum(1 for v in self._sfm.views if v.registered)
            if self._sfm else 0,
            "n_points_3d": len(self._sfm.points_3d) if self._sfm else 0,
            "timings_ms": self._timings,
        }
        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"结果已导出到 {self.output_dir}/")

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _load_images(directory: Path, glob_pattern: str) -> List[np.ndarray]:
        """从 *directory* 中加载匹配 *glob_pattern* 的图像。"""
        if not directory.exists():
            return []
        paths = sorted(directory.glob(glob_pattern))
        images = []
        for p in paths:
            img = cv2.imread(str(p))
            if img is not None:
                images.append(img)
        return images
