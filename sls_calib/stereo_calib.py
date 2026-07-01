"""
双目相机标定 —— 双相机标定与立体校正。

利用两台相机对同一标定靶的同时观测，计算它们之间的
刚体变换 (R, T)，然后生成用于极线对齐立体图像对的
校正映射。

支持多种标定图案类型：
  - 棋盘格          (cv2.findChessboardCorners)
  - 圆形网格        (SLSMarkerDetector + 网格分配)
  - ChArUco 板      (CodedMarkerDetector + cv2.aruco)

典型工作流程
--------------
  1. 单独标定每台相机       →  K_left, dist_left,
                                 K_right, dist_right
  2. 加载立体图像对         →  List[(imgL, imgR), …]
  3. 在所有图像对中检测图案 →  object_points,
                                 left_image_points,
                                 right_image_points
  4. stereoCalibrate         →  R, T, E, F
  5. stereoRectify           →  R1, R2, P1, P2, Q
  6. initUndistortRectifyMap →  重映射查找表
  7. remap                   →  校正后的立体图像对

参考文献
--------
  OpenCV: cv2.stereoCalibrate, cv2.stereoRectify,
          cv2.initUndistortRectifyMap, cv2.remap
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .marker_detector import SLSMarkerDetector
from .camera_calib import CalibImage, Calibrator

# ---------------------------------------------------------------------------
# 类型别名
# ---------------------------------------------------------------------------


@dataclass
class StereoParams:
    """立体标定结果（包含立体校正所需的全部数据）。"""

    K_left: np.ndarray        # 3×3  左相机内参矩阵
    dist_left: np.ndarray     #      左相机畸变系数
    K_right: np.ndarray       # 3×3  右相机内参矩阵
    dist_right: np.ndarray    #      右相机畸变系数

    R: np.ndarray             # 3×3  旋转：右相机 → 左相机
    T: np.ndarray             # 3×1  平移：右相机在左相机坐标系下
    E: np.ndarray             # 3×3  本质矩阵
    F: np.ndarray             # 3×3  基础矩阵

    R1: np.ndarray            # 3×3  左相机校正旋转
    R2: np.ndarray            # 3×3  右相机校正旋转
    P1: np.ndarray            # 3×4  左相机校正后投影矩阵
    P2: np.ndarray            # 3×4  右相机校正后投影矩阵
    Q: np.ndarray             # 4×4  视差-深度映射矩阵

    image_size: Tuple[int, int]  # (宽, 高)

    rms_error: float = 0.0    # 立体标定 RMS 重投影误差


# ===================================================================
# StereoCalibrator
# ===================================================================


class StereoCalibrator:
    """
    双目相机标定与立体校正。

    处理完整流水线：
      内参标定（可选） → 立体标定 → 立体校正。

    图案类型
    --------
    ``"chessboard"``
        标准黑白棋盘格。默认选项，使用最简单。
        *pattern_size* 为内角点的 ``(cols, rows)``，
        *square_size* 为方格边长（米）。

    ``"circles"``
        SLS 风格的圆形点靶标（与 ``Calibrator`` 使用的
        11×9 网格相同）。需要圆点检测器和网格逻辑。

    ``"charuco"``
        ChArUco 板（棋盘格 + ArUco）。最鲁棒 ——
        ArUco 标志点提供自动 ID 分配，即使部分遮挡也能工作。
        *pattern_size* = ``(squares_x, squares_y)``。
    """

    _PATTERN_CHESS = "chessboard"
    _PATTERN_CIRCLES = "circles"
    _PATTERN_CHARUCO = "charuco"

    def __init__(
        self,
        pattern_type: str = "chessboard",
        pattern_size: Tuple[int, int] = (9, 6),
        square_size: float = 0.025,
        aruco_dict_name: str = "4x4_50",
        marker_size: float = 0.015,
    ) -> None:
        """
        Args:
            pattern_type: ``"chessboard"`` | ``"circles"`` | ``"charuco"``。
            pattern_size: 内角点或方格的 ``(cols, rows)``。
            square_size: 单个方格的物理边长（米）。
            aruco_dict_name: ArUco 字典（用于 ChArUco）。
            marker_size: ArUco 标志点的物理边长（米）。
        """
        if pattern_type not in (self._PATTERN_CHESS,
                                self._PATTERN_CIRCLES,
                                self._PATTERN_CHARUCO):
            raise ValueError(
                f"未知的 pattern_type '{pattern_type}'。"
                f"请使用 'chessboard'、'circles' 或 'charuco'。"
            )

        self.pattern_type = pattern_type
        self.pattern_size = pattern_size
        self.square_size = square_size
        self.marker_size = marker_size
        self.aruco_dict_name = aruco_dict_name

        # 内部检测器（对 circles/charuco 延迟初始化）
        self._circle_detector: Optional[SLSMarkerDetector] = None
        self._calibrator: Optional[Calibrator] = None
        self._charuco_board: Optional[cv2.aruco.CharucoBoard] = None
        self._aruco_detector: Optional[cv2.aruco.ArucoDetector] = None

        # 缓存图案的三维物方点
        self._obj_points_cache: Optional[np.ndarray] = None

        # 标定结果
        self._K_left: Optional[np.ndarray] = None
        self._dist_left: Optional[np.ndarray] = None
        self._K_right: Optional[np.ndarray] = None
        self._dist_right: Optional[np.ndarray] = None
        self._stereo_params: Optional[StereoParams] = None

        # 立体校正映射
        self._map_left_x: Optional[np.ndarray] = None
        self._map_left_y: Optional[np.ndarray] = None
        self._map_right_x: Optional[np.ndarray] = None
        self._map_right_y: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # 标定图案的三维物方点
    # ------------------------------------------------------------------

    def _get_object_points(self) -> np.ndarray:
        """构建标定板局部坐标系中的 (N, 3) 图案点数组。"""
        if self._obj_points_cache is not None:
            return self._obj_points_cache

        if self.pattern_type == self._PATTERN_CHESS:
            cols, rows = self.pattern_size
            pts = np.zeros((rows * cols, 3), dtype=np.float32)
            pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
            pts *= self.square_size

        elif self.pattern_type == self._PATTERN_CIRCLES:
            # 11×9 网格，5 个大基准圆
            # 这是 SLS 标定板的布局
            pts_list = []
            for row in range(9):
                for col in range(11):
                    # 大圆位于特定的网格位置
                    large_positions = {(5, 2), (2, 4), (8, 4), (5, 6), (6, 6)}
                    pts_list.append([col, row, 0.0])
            pts = np.array(pts_list, dtype=np.float32)
            pts[:, :2] *= self.square_size

        elif self.pattern_type == self._PATTERN_CHARUCO:
            sx, sy = self.pattern_size
            board = self._get_charuco_board()
            # ChArUco 板的物方点为棋盘格角点
            pts = np.array([
                [c * self.square_size, r * self.square_size, 0.0]
                for r in range(sy - 1) for c in range(sx - 1)
            ], dtype=np.float32)

        else:
            raise RuntimeError(f"未知图案: {self.pattern_type}")

        self._obj_points_cache = pts
        return pts

    # ------------------------------------------------------------------
    # 图案检测
    # ------------------------------------------------------------------

    def detect_pattern(
        self,
        image: np.ndarray,
        debug: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        """
        在单张图像中检测标定图案点。

        Returns:
            ``(corners, object_points, error)``。
            *corners* 为 ``(N, 1, 2)`` 的浮点二维点数组
            （OpenCV 约定），失败时为 ``None``。
            *object_points* 为对应三维坐标的 ``(N, 1, 3)`` 数组。
        """
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.pattern_type == self._PATTERN_CHESS:
            return self._detect_chessboard(gray, debug)

        elif self.pattern_type == self._PATTERN_CIRCLES:
            return self._detect_circles(image, debug)  # 检测器需要 BGR

        elif self.pattern_type == self._PATTERN_CHARUCO:
            return self._detect_charuco(gray, debug)

        return None, None, f"未知图案: {self.pattern_type}"

    def _detect_chessboard(
        self, gray: np.ndarray, debug: bool
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        cols, rows = self.pattern_size
        found, corners = cv2.findChessboardCornersSB(
            gray, (cols, rows),
            flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY,
        )
        if not found:
            return None, None, "未找到棋盘格"

        # 亚像素精化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

        obj_pts = self._get_object_points()

        return corners, obj_pts.reshape(-1, 1, 3), ""

    def _detect_circles(
        self, image: np.ndarray, debug: bool
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        # 延迟初始化圆形检测器和标定器
        if self._circle_detector is None:
            self._circle_detector = SLSMarkerDetector()
        if self._calibrator is None:
            self._calibrator = Calibrator()

        calib_img = CalibImage(name="stereo_pair", image=image)
        markers, err = self._circle_detector.detectMarkers(
            image, smooth=False, debug=debug
        )
        if err:
            return None, None, err
        if not markers:
            return None, None, "未检测到圆"

        calib_img.circles = markers
        calib_img.create_display_circles()
        err2 = calib_img.find_circle_indices(self.square_size, debug=debug)
        if err2:
            return None, None, err2

        # 从 circle_array 中提取有效的 2D-3D 对应关系
        img_pts = []
        obj_pts = []
        for (px, py), (wx, wy, wz), valid, _ in calib_img.circle_array:
            if valid:
                img_pts.append([px, py])
                obj_pts.append([wx, wy, wz])

        if not img_pts:
            return None, None, "没有有效的网格分配"

        return (
            np.array(img_pts, dtype=np.float32).reshape(-1, 1, 2),
            np.array(obj_pts, dtype=np.float32).reshape(-1, 1, 3),
            "",
        )

    def _detect_charuco(
        self, gray: np.ndarray, debug: bool
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        board = self._get_charuco_board()
        detector = self._get_aruco_detector()

        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None or len(ids) < 4:
            return None, None, "ArUco 标志点太少，无法用于 ChArUco"

        n_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board
        )
        if charuco_corners is None or n_corners < 4:
            return None, None, "ChArUco 角点插值失败"

        obj_pts, img_pts = cv2.aruco.getBoardObjectAndImagePoints(
            board, charuco_corners, charuco_ids
        )
        return img_pts, obj_pts, ""

    # ------------------------------------------------------------------
    # 内参标定（单相机）
    # ------------------------------------------------------------------

    def calibrate_intrinsics(
        self,
        images: List[np.ndarray],
        image_size: Optional[Tuple[int, int]] = None,
        debug: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        使用多个图案视图标定一台相机。

        Args:
            images: 包含同一标定图案的图像列表。
            image_size: ``(width, height)`` —— 如果为 None 则自动检测。

        Returns:
            ``(K, dist_coeffs, rms_error)``，失败时返回 ``(None, None, inf)``。
        """
        if not images:
            return None, None, float("inf")

        if image_size is None:
            h, w = images[0].shape[:2]
            image_size = (w, h)

        obj_pts_all = []
        img_pts_all = []

        for img in images:
            corners, obj_pts, err = self.detect_pattern(img, debug=debug)
            if corners is not None:
                obj_pts_all.append(obj_pts.reshape(-1, 3))
                img_pts_all.append(corners.reshape(-1, 2))

        if len(obj_pts_all) < 3:
            print(f"  仅有 {len(obj_pts_all)} 张有效图像（需要 >= 3）。")
            return None, None, float("inf")

        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_pts_all, img_pts_all, image_size,
            None, None,
            flags=cv2.CALIB_FIX_K3,
        )

        return K, dist.ravel(), ret

    # ------------------------------------------------------------------
    # 立体标定
    # ------------------------------------------------------------------

    def calibrate_stereo_from_corners(
        self,
        left_corners: List[np.ndarray],
        right_corners: List[np.ndarray],
        object_points: np.ndarray,
        K_left: np.ndarray,
        dist_left: np.ndarray,
        K_right: np.ndarray,
        dist_right: np.ndarray,
        image_size: Tuple[int, int],
        fix_intrinsics: bool = True,
        debug: bool = False,
    ) -> Optional[StereoParams]:
        """
        使用预先检测到的角点坐标标定立体相机系统。

        这是核心标定流程 —— 它跳过图案检测，直接使用
        2D↔3D 对应关系进行标定。当你已拥有角点位置时
        （例如来自合成数据或外部检测器）使用此方法。

        Args:
            left_corners:  二维角点位置的 ``(N,1,2)`` 数组列表。
            right_corners: 二维角点位置的 ``(N,1,2)`` 数组列表。
            object_points: 三维标定板点的 ``(N,3)`` 数组。
            K_left, dist_left, K_right, dist_right: 内参。
            image_size: ``(width, height)``。
            fix_intrinsics: 保持内参不变（推荐）。

        Returns:
            ``StereoParams``，失败时返回 ``None``。
        """
        n = len(left_corners)
        if n < 5:
            print(f"需要 >= 5 对图像（当前 {n} 对）。")
            return None

        obj_pts_all = [
            object_points.reshape(-1, 1, 3).astype(np.float32)
        ] * n
        img_left = [
            c.reshape(-1, 1, 2).astype(np.float32) for c in left_corners
        ]
        img_right = [
            c.reshape(-1, 1, 2).astype(np.float32) for c in right_corners
        ]

        flags = cv2.CALIB_FIX_INTRINSIC if fix_intrinsics else 0
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100, 1e-5,
        )

        ret, K1, d1, K2, d2, R, T, E, F = cv2.stereoCalibrate(
            obj_pts_all, img_left, img_right,
            K_left.astype(np.float64), dist_left.astype(np.float64),
            K_right.astype(np.float64), dist_right.astype(np.float64),
            image_size, criteria=criteria, flags=flags,
        )

        if debug:
            print(f"  stereoCalibrate RMS = {ret:.4f} px")

        # 立体校正
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K1, d1, K2, d2, image_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
        )

        self._stereo_params = StereoParams(
            K_left=K1, dist_left=d1.ravel(),
            K_right=K2, dist_right=d2.ravel(),
            R=R, T=T, E=E, F=F,
            R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
            image_size=image_size, rms_error=ret,
        )

        self._K_left, self._dist_left = K1, d1.ravel()
        self._K_right, self._dist_right = K2, d2.ravel()
        self._build_rectification_maps()

        return self._stereo_params

    def calibrate_stereo(
        self,
        left_images: List[np.ndarray],
        right_images: List[np.ndarray],
        K_left: np.ndarray,
        dist_left: np.ndarray,
        K_right: np.ndarray,
        dist_right: np.ndarray,
        image_size: Optional[Tuple[int, int]] = None,
        fix_intrinsics: bool = True,
        debug: bool = False,
    ) -> Optional[StereoParams]:
        """
        标定立体相机系统：计算两相机之间的 R, T, E, F。

        在每个图像对中检测标定图案，然后调用
        ``calibrate_stereo_from_corners``。

        Returns:
            ``StereoParams``，失败时返回 ``None``。
        """
        if len(left_images) != len(right_images):
            raise ValueError(
                f"图像数量不匹配："
                f"左 {len(left_images)} 张 vs 右 {len(right_images)} 张"
            )
        if len(left_images) < 5:
            print(f"需要 >= 5 对立体图像（当前 {len(left_images)} 对）。")
            return None

        if image_size is None:
            h, w = left_images[0].shape[:2]
            image_size = (w, h)

        obj_pts_all: List[np.ndarray] = []
        img_pts_left_all: List[np.ndarray] = []
        img_pts_right_all: List[np.ndarray] = []

        t0 = time.perf_counter()

        for idx, (imgL, imgR) in enumerate(zip(left_images, right_images)):
            cornersL, objL, errL = self.detect_pattern(imgL, debug=False)
            cornersR, objR, errR = self.detect_pattern(imgR, debug=False)

            if cornersL is None or cornersR is None:
                if debug:
                    print(f"  对 {idx}：跳过 "
                          f"(L={'OK' if cornersL is not None else errL}, "
                          f"R={'OK' if cornersR is not None else errR})")
                continue

            obj_pts_all.append(objL.reshape(-1, 3))
            img_pts_left_all.append(cornersL.reshape(-1, 2))
            img_pts_right_all.append(cornersR.reshape(-1, 2))

        n_valid = len(obj_pts_all)
        if n_valid < 5:
            print(f"仅有 {n_valid} 对有效图像（需要 >= 5）。")
            return None

        if debug:
            print(f"  {n_valid}/{len(left_images)} 对有效 "
                  f"({(time.perf_counter() - t0) * 1000:.0f} ms)")

        return self.calibrate_stereo_from_corners(
            [c.reshape(-1, 1, 2) for c in img_pts_left_all],
            [c.reshape(-1, 1, 2) for c in img_pts_right_all],
            obj_pts_all[0],
            K_left, dist_left, K_right, dist_right,
            image_size, fix_intrinsics, debug,
        )

    # ------------------------------------------------------------------
    # 立体校正
    # ------------------------------------------------------------------

    def _build_rectification_maps(self) -> None:
        """预计算用于快速立体校正的重映射查找表。"""
        if self._stereo_params is None:
            return

        sp = self._stereo_params
        self._map_left_x, self._map_left_y = cv2.initUndistortRectifyMap(
            sp.K_left, sp.dist_left, sp.R1, sp.P1,
            sp.image_size, cv2.CV_32FC1,
        )
        self._map_right_x, self._map_right_y = cv2.initUndistortRectifyMap(
            sp.K_right, sp.dist_right, sp.R2, sp.P2,
            sp.image_size, cv2.CV_32FC1,
        )

    def rectify(
        self, left_image: np.ndarray, right_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        校正一对立体图像。

        返回 ``(rectified_left, rectified_right)``，二者对齐使得
        对应的极线为水平线。
        """
        if self._map_left_x is None:
            raise RuntimeError("请先运行 calibrate_stereo()。")

        left_rect = cv2.remap(
            left_image,
            self._map_left_x, self._map_left_y,
            cv2.INTER_LINEAR,
        )
        right_rect = cv2.remap(
            right_image,
            self._map_right_x, self._map_right_y,
            cv2.INTER_LINEAR,
        )
        return left_rect, right_rect

    def compute_disparity(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        rectify_first: bool = True,
        num_disparities: int = 128,
        block_size: int = 15,
    ) -> np.ndarray:
        """
        使用 SGBM 算法计算立体图像对的视差图。

        Args:
            left_image, right_image: 立体图像对。
            rectify_first: 匹配前先进行立体校正。
            num_disparities: 必须能被 16 整除。
            block_size: 必须为奇数。

        Returns:
            视差图（float32），缩放因子为 16.0
            （除以 16 得到像素视差值）。
            使用 ``stereo_params.Q`` 可转换为深度。
        """
        if rectify_first:
            left, right = self.rectify(left_image, right_image)
        else:
            left, right = left_image, right_image

        gray_l = left if left.ndim == 2 else cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_r = right if right.ndim == 2 else cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        stereo_sgbm = cv2.StereoSGBM.create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * 3 * block_size ** 2,
            P2=32 * 3 * block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
        )

        disparity = stereo_sgbm.compute(gray_l, gray_r).astype(np.float32) / 16.0
        return disparity

    def disparity_to_depth(
        self, disparity: np.ndarray
    ) -> np.ndarray:
        """
        使用 Q 矩阵将视差图转换为深度图（米）。

        返回深度图（float32），单位为米，无效区域为 0。
        """
        if self._stereo_params is None:
            raise RuntimeError("请先运行 calibrate_stereo()。")

        points_3d = cv2.reprojectImageTo3D(
            disparity, self._stereo_params.Q, handleMissingValues=True
        )
        depth = points_3d[:, :, 2]  # Z 通道
        return depth.astype(np.float32)

    # ------------------------------------------------------------------
    # 延迟初始化辅助方法
    # ------------------------------------------------------------------

    def _get_charuco_board(self) -> cv2.aruco.CharucoBoard:
        if self._charuco_board is None:
            from sls_calib.coded_marker import CodedMarkerDetector
            dict_id = CodedMarkerDetector._DICT_MAP[self.aruco_dict_name]
            aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
            sx, sy = self.pattern_size
            self._charuco_board = cv2.aruco.CharucoBoard(
                (sx, sy), self.square_size, self.marker_size, aruco_dict,
            )
        return self._charuco_board

    def _get_aruco_detector(self) -> cv2.aruco.ArucoDetector:
        if self._aruco_detector is None:
            from sls_calib.coded_marker import CodedMarkerDetector
            dict_id = CodedMarkerDetector._DICT_MAP[self.aruco_dict_name]
            aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
            params = cv2.aruco.DetectorParameters()
            self._aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        return self._aruco_detector

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def stereo_params(self) -> Optional[StereoParams]:
        """最近一次 ``calibrate_stereo()`` 调用的结果。"""
        return self._stereo_params

    @property
    def baseline(self) -> Optional[float]:
        """立体基线长度，单位为**米**（T 的范数）。"""
        if self._stereo_params is None:
            return None
        return float(np.linalg.norm(self._stereo_params.T))

    @property
    def is_calibrated(self) -> bool:
        """立体标定是否已完成。"""
        return self._stereo_params is not None

    # ------------------------------------------------------------------
    # 摘要
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """人类可读的立体标定摘要。"""
        if self._stereo_params is None:
            return "StereoCalibrator：尚未标定。"

        sp = self._stereo_params
        lines = [
            "立体标定摘要",
            f"  图案：      {self.pattern_type} {self.pattern_size}",
            f"  图像尺寸：  {sp.image_size[0]}×{sp.image_size[1]}",
            f"  RMS 误差：  {sp.rms_error:.4f} px",
            f"  基线：      {self.baseline * 1000:.2f} mm",
            "",
            "  左相机：",
            f"    K = [{sp.K_left[0,0]:.1f}, {sp.K_left[1,1]:.1f}] "
            f"cx,cy=({sp.K_left[0,2]:.1f}, {sp.K_left[1,2]:.1f})",
            f"    dist = {np.array2string(sp.dist_left, precision=4, suppress_small=True)}",
            "",
            "  右相机：",
            f"    K = [{sp.K_right[0,0]:.1f}, {sp.K_right[1,1]:.1f}] "
            f"cx,cy=({sp.K_right[0,2]:.1f}, {sp.K_right[1,2]:.1f})",
            f"    dist = {np.array2string(sp.dist_right, precision=4, suppress_small=True)}",
            "",
            "  立体变换（右相机在左相机坐标系下）：",
            f"    R = {np.array2string(sp.R, precision=4, suppress_small=True)}",
            f"    T = {np.array2string(sp.T.ravel(), precision=4, suppress_small=True)}  (m)",
        ]
        return "\n".join(lines)


# ===================================================================
# 便捷函数：单次调用完成立体标定（内参 + 外参）
# ===================================================================


def calibrate_stereo_rig(
    left_images: List[np.ndarray],
    right_images: List[np.ndarray],
    pattern_type: str = "chessboard",
    pattern_size: Tuple[int, int] = (9, 6),
    square_size: float = 0.025,
    debug: bool = False,
) -> Optional[StereoCalibrator]:
    """
    一键立体标定：内参 → 外参 → 立体校正。

    便捷封装函数，依次运行每台相机的内参标定和立体标定，
    一次调用完成全部流程。

    Args:
        left_images, right_images: 标定靶的配对图像。
            必须一一对应（left_images[i] ↔ right_images[i]）。
        pattern_type: ``"chessboard"`` | ``"circles"`` | ``"charuco"``。
        pattern_size: 内角点的 ``(cols, rows)``。
        square_size: 方格物理边长，单位为**米**。
        debug: 打印详细进度。

    Returns:
        标定完成的 ``StereoCalibrator``，失败时返回 ``None``。
    """
    if len(left_images) != len(right_images):
        raise ValueError(
            f"图像数量不匹配：{len(left_images)} vs {len(right_images)}"
        )

    calib = StereoCalibrator(
        pattern_type=pattern_type,
        pattern_size=pattern_size,
        square_size=square_size,
    )

    # --- 内参标定 -----------------------------------------------
    if debug:
        print("--- 左相机内参 ---")
    K_left, dist_left, rms_left = calib.calibrate_intrinsics(
        left_images, debug=debug
    )
    if K_left is None:
        print("左相机内参标定失败。")
        return None
    if debug:
        print(f"  RMS = {rms_left:.4f} px")

    if debug:
        print("--- 右相机内参 ---")
    K_right, dist_right, rms_right = calib.calibrate_intrinsics(
        right_images, debug=debug
    )
    if K_right is None:
        print("右相机内参标定失败。")
        return None
    if debug:
        print(f"  RMS = {rms_right:.4f} px")

    # --- 立体标定 ------------------------------------------------
    if debug:
        print("--- 立体标定 ---")
    sp = calib.calibrate_stereo(
        left_images, right_images,
        K_left, dist_left,
        K_right, dist_right,
        debug=debug,
    )
    if sp is None:
        print("立体标定失败。")
        return None

    if debug:
        print()
        print(calib.summary())

    return calib


# ===================================================================
# 合成数据测试
# ===================================================================


def _generate_synthetic_stereo_data(
    n_pairs: int = 10,
    image_size: Tuple[int, int] = (1280, 720),
    pattern_size: Tuple[int, int] = (9, 6),
    square_size: float = 0.025,
    noise_px: float = 0.3,
) -> Tuple[
    List[np.ndarray],       # 左图像列表
    List[np.ndarray],       # 右图像列表
    List[np.ndarray],       # 投影角点（左）
    List[np.ndarray],       # 投影角点（右）
    np.ndarray,             # 物方点 (N×3)
    np.ndarray,             # K_left
    np.ndarray,             # dist_left
    np.ndarray,             # K_right
    np.ndarray,             # dist_right
    np.ndarray,             # GT R（右相机在左相机坐标系下）
    np.ndarray,             # GT T
]:
    """
    为棋盘格靶标生成合成立体图像对。

    同时返回渲染后的图像（用于可视化）和投影后的角点坐标
    （可直接注入标定流程，跳过棋盘格检测）。
    这是测试标定数学逻辑的标准方法。
    """
    rng = np.random.default_rng(42)
    w, h = image_size
    cols, rows = pattern_size

    # 真实的相机内参
    fx = 800.0
    K = np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)

    K_left = K.copy()
    dist_left = dist.copy()
    K_right = K.copy()
    dist_right = dist.copy()

    # 立体基线：右相机沿 +X 方向偏移
    baseline = 0.12  # 12 cm
    R_gt = np.eye(3)
    T_gt = np.array([baseline, 0.0, 0.0]).reshape(3, 1)

    # 三维棋盘格角点（在标定板局部坐标系中，z=0）
    obj_pts = np.zeros((rows * cols, 3), dtype=np.float32)
    obj_pts[:, :2] = (
        np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
    )

    left_images, right_images = [], []
    left_corners_list, right_corners_list = [], []

    attempts = 0
    max_attempts = n_pairs * 10

    while len(left_images) < n_pairs and attempts < max_attempts:
        attempts += 1
        imgL = np.full((h, w, 3), 240, dtype=np.uint8)
        imgR = np.full((h, w, 3), 240, dtype=np.uint8)

        # 标定板位于约 0.3–0.6 m，居中，适度倾斜
        board_z = 0.3 + rng.random() * 0.3
        board_x = (rng.random() - 0.5) * 0.15
        board_y = (rng.random() - 0.5) * 0.12
        board_pos = np.array([board_x, board_y, board_z])

        tilt_x = (rng.random() - 0.5) * np.deg2rad(40)
        tilt_y = (rng.random() - 0.5) * np.deg2rad(40)
        tilt_z = (rng.random() - 0.5) * np.deg2rad(15)
        R_board = cv2.Rodrigues(np.array([tilt_x, tilt_y, tilt_z]))[0]

        rvec_board = cv2.Rodrigues(R_board)[0]
        tvec_board = -R_board @ board_pos.astype(np.float64)

        # 投影到左相机
        projL, _ = cv2.projectPoints(
            obj_pts, rvec_board, tvec_board, K_left, dist_left
        )
        # 右相机：X_right_cam = R_gt @ X_left_cam + T_gt
        #          = R_gt @ (R_board @ X_obj + tvec) + T_gt
        R_right = R_gt @ R_board
        rvec_right = cv2.Rodrigues(R_right)[0]
        tvec_right = (R_gt @ tvec_board.ravel() + T_gt.ravel()).astype(np.float64)
        projR, _ = cv2.projectPoints(
            obj_pts, rvec_right, tvec_right.reshape(3, 1),
            K_right, dist_right,
        )

        # 检查是否在图像边界内（含边距）
        projL_2d = projL.reshape(-1, 2)
        projR_2d = projR.reshape(-1, 2)
        margin = 40
        if not (
            np.all((projL_2d > margin) & (projL_2d < np.array([w - margin, h - margin])))
            and np.all((projR_2d > margin) & (projR_2d < np.array([w - margin, h - margin])))
        ):
            continue

        # 向角点坐标添加噪声并绘制
        noisyL = projL_2d + rng.normal(0, noise_px, projL_2d.shape)
        noisyR = projR_2d + rng.normal(0, noise_px, projR_2d.shape)

        for ptL, ptR in zip(noisyL, noisyR):
            cv2.circle(imgL, tuple(ptL.astype(int)), 3, (0, 0, 0), -1)
            cv2.circle(imgR, tuple(ptR.astype(int)), 3, (0, 0, 0), -1)

        left_images.append(imgL)
        right_images.append(imgR)
        left_corners_list.append(noisyL.astype(np.float32).reshape(-1, 1, 2))
        right_corners_list.append(noisyR.astype(np.float32).reshape(-1, 1, 2))

    return (
        left_images, right_images,
        left_corners_list, right_corners_list,
        obj_pts,
        K_left, dist_left, K_right, dist_right,
        R_gt, T_gt,
    )


# ===================================================================
# （测试代码已提取到 tests/test_stereo_calib.py）
# ===================================================================
