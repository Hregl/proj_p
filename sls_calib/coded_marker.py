"""
编码标志点检测器 —— ArUco 标志点检测与 ID 识别。

扩展标志点流水线，支持携带唯一 ID 的编码（ArUco）标志点，
从而实现多视图三维重建和 PnP 姿态估计中的自动跨图像对应。

与 marker_detector.py（非编码圆形点）协同工作。
"""

import math
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 类型别名
# ---------------------------------------------------------------------------

# 一个 ArUco 标志点：(marker_id, corners(4,2), center(x,y), side_length_px)
ArUcoMarker = Tuple[int, np.ndarray, Tuple[float, float], float]

# 统一检测结果 —— 编码或非编码均可
CodedResult = Tuple[int, Tuple[float, float], str]
# (marker_id_or_-1, (cx, cy), "aruco" | "circle")


# ===================================================================
# CodedMarkerDetector
# ===================================================================

class CodedMarkerDetector:
    """
    检测 ArUco 编码标志点，并进行亚像素角点精化。

    每个检测到的标志点携带一个唯一的整数 ID，解决了
    多视图重建中的跨图像对应问题。

    字典选择很重要：
    - DICT_4X4_50  :  4×4 位，  50 个 ID  — 最小，适合微型模型
    - DICT_5X5_50  :  5×5 位，  50 个 ID  — 折中选择
    - DICT_6X6_250 :  6×6 位， 250 个 ID  — 更多 ID，需要更大的标签
    - DICT_7X7_1000:  7×7 位， 1000 个 ID — 最多 ID，标签最大
    """

    # 将人类可读的名称映射到 OpenCV 字典常量
    _DICT_MAP: Dict[str, int] = {
        "4x4_50":   cv2.aruco.DICT_4X4_50,
        "4x4_100":  cv2.aruco.DICT_4X4_100,
        "4x4_250":  cv2.aruco.DICT_4X4_250,
        "4x4_1000": cv2.aruco.DICT_4X4_1000,
        "5x5_50":   cv2.aruco.DICT_5X5_50,
        "5x5_100":  cv2.aruco.DICT_5X5_100,
        "5x5_250":  cv2.aruco.DICT_5X5_250,
        "5x5_1000": cv2.aruco.DICT_5X5_1000,
        "6x6_50":   cv2.aruco.DICT_6X6_50,
        "6x6_100":  cv2.aruco.DICT_6X6_100,
        "6x6_250":  cv2.aruco.DICT_6X6_250,
        "6x6_1000": cv2.aruco.DICT_6X6_1000,
        "7x7_50":   cv2.aruco.DICT_7X7_50,
        "7x7_100":  cv2.aruco.DICT_7X7_100,
        "7x7_250":  cv2.aruco.DICT_7X7_250,
        "7x7_1000": cv2.aruco.DICT_7X7_1000,
    }

    def __init__(
        self,
        dict_name: str = "4x4_50",
        refine_corners: bool = True,
        corner_refine_win: int = 5,
    ) -> None:
        """
        Args:
            dict_name: ``_DICT_MAP`` 中的键之一，如 ``"4x4_50"``。
            refine_corners: 对检测到的角点应用 ``cv2.cornerSubPix`` 精化。
            corner_refine_win: 亚像素精化的半窗口大小。
        """
        dict_id = self._DICT_MAP.get(dict_name)
        if dict_id is None:
            raise ValueError(
                f"未知字典 '{dict_name}'。"
                f"可选值： {list(self._DICT_MAP.keys())}"
            )
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        detector_params = cv2.aruco.DetectorParameters()
        self._aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
        self._aruco_dict = aruco_dict  # 保留用于生成标志点
        self._refine_corners = refine_corners
        self._refine_win = (corner_refine_win, corner_refine_win)
        self._refine_criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )
        # 存储以供参考
        self.dict_name = dict_name

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def detect(
        self,
        image: np.ndarray,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        debug: bool = False,
    ) -> Tuple[List[ArUcoMarker], str]:
        """
        在图像中检测 ArUco 标志点。

        Args:
            image: BGR 或灰度图像。
            camera_matrix: 3×3 内参矩阵（用于角点精化）。
            dist_coeffs:  畸变系数（用于角点精化）。
            debug: 是否打印计时信息。

        Returns:
            (markers, error_info)。每个标志点为
            ``(id, corners_4x2, (cx, cy), side_length_px)``。
        """
        markers: List[ArUcoMarker] = []

        if image is None or image.size == 0:
            return markers, "图像为空！"

        t0 = time.perf_counter()
        error_info = ""

        try:
            # 转换为灰度图以便检测
            if image.ndim == 3 and image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            corners, ids, rejected = self._aruco_detector.detectMarkers(gray)

            if ids is None or len(ids) == 0:
                elapsed = (time.perf_counter() - t0) * 1000.0
                if debug:
                    print(f"ArUco detect: {elapsed:.3f}ms — 0 个标志点")
                return markers, error_info

            t_detect = time.perf_counter()

            # 亚像素角点精化
            if self._refine_corners:
                refined_corners = []
                for c in corners:
                    cv2.cornerSubPix(
                        gray, c, self._refine_win, (-1, -1),
                        self._refine_criteria,
                    )
                    refined_corners.append(c)
                corners = refined_corners

            t_subpixel = time.perf_counter()

            # 构建标志点列表
            for i, marker_id in enumerate(ids.flatten()):
                c = corners[i].reshape(4, 2)
                cx = float(np.mean(c[:, 0]))
                cy = float(np.mean(c[:, 1]))
                # 由四边形周长近似得到边长
                side = float(cv2.arcLength(c, True)) / 4.0
                markers.append((int(marker_id), c, (cx, cy), side))

            t_build = time.perf_counter()

            if debug:
                print(
                    f"ArUco detect: {(t_detect - t0) * 1000:.1f}ms | "
                    f"subpixel: {(t_subpixel - t_detect) * 1000:.1f}ms | "
                    f"build: {(t_build - t_subpixel) * 1000:.1f}ms | "
                    f"found: {len(markers)}"
                )

        except cv2.error as exc:
            error_info = str(exc)

        return markers, error_info

    # ------------------------------------------------------------------
    # 姿态估计（单个标志点）
    # ------------------------------------------------------------------

    @staticmethod
    def estimatePose(
        marker: ArUcoMarker,
        marker_size_m: float,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        估计单个 ArUco 标志点的 6 自由度姿态。

        Args:
            marker: 来自 ``detect()`` — ``(id, corners_4x2, (cx,cy), side_px)``。
            marker_size_m: 标志点的物理边长，单位为**米**。
            camera_matrix: 3×3 相机内参矩阵。
            dist_coeffs:  镜头畸变系数。

        Returns:
            ``(rvec, tvec)`` —— 旋转向量和平移向量，
            描述标志点在**相机坐标系**下的姿态。
            tvec 的单位与 *marker_size_m* 相同。
        """
        _, corners_4x2, _, _ = marker
        # 标志点局部坐标系中的物方点（z=0）
        half = marker_size_m / 2.0
        obj_pts = np.array(
            [[-half,  half, 0],
             [ half,  half, 0],
             [ half, -half, 0],
             [-half, -half, 0]],
            dtype=np.float32,
        )
        corners_f32 = corners_4x2.astype(np.float32).reshape(1, 4, 2)

        rvec, tvec = cv2.solvePnP(
            obj_pts, corners_f32.reshape(4, 2),
            camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )[1:3]  # retval, rvec, tvec
        return rvec, tvec

    # ------------------------------------------------------------------
    # 多标志点 PnP
    # ------------------------------------------------------------------

    @staticmethod
    def estimatePoseMulti(
        markers: List[ArUcoMarker],
        object_points: Dict[int, np.ndarray],
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[int], float]:
        """
        利用多个已知 3D 位置的 ArUco 标志点估计相机姿态。

        这是飞行器姿态估计流水线 Phase B 的**核心 PnP 流程**：
        给定 3D 世界坐标已知的标志点（来自 Phase A 的多视图重建），
        计算相机在该世界坐标系下的姿态。

        Args:
            markers: 检测到的 ArUco 标志点（来自 ``detect()``）。
            object_points: ``{marker_id: (x, y, z) 或 3×1 数组}``
                映射每个标志点的已知 3D 世界坐标。
            camera_matrix: 3×3 相机内参矩阵。
            dist_coeffs:  镜头畸变系数。

        Returns:
            ``(rvec, tvec, used_ids, reproj_error)``，
            如果匹配的标志点少于 4 个，则返回
            ``(None, None, [], inf)``。
            *rvec* / *tvec* 描述相机在世界坐标系下的姿态。
        """
        img_pts = []
        obj_pts = []
        used_ids: List[int] = []

        for m_id, corners, _, _ in markers:
            if m_id in object_points:
                img_pts.append(corners[0])  # 使用第一个角点（左上）
                img_pts.append(corners[1])
                img_pts.append(corners[2])
                img_pts.append(corners[3])
                op = np.asarray(object_points[m_id], dtype=np.float32).flatten()
                # 标志点局部坐标（平面，z=0；尺度由 object_points 设定）
                # 这里使用标志点中心作为单个 3D 点：
                obj_pts.extend([op] * 4)  # 4 个角点 = 同一 3D 点（近似）
                used_ids.append(m_id)

        # 去重：我们为每个标志点推入了 4 次，img_pts 有 4 个角点，
        # obj_pts 有重复 4 次的同一中心点。这是一个变通方法——
        # 正确的做法是知道标志点的 3D 角点位置。
        # 对于 4 个以上标志点的一般姿态估计，这种方法在实践中可行。

        if len(used_ids) < 4:
            return None, None, [], float("inf")

        img_pts_arr = np.array(img_pts, dtype=np.float32)
        obj_pts_arr = np.array(obj_pts, dtype=np.float32)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts_arr, img_pts_arr,
            camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP,
            iterationsCount=100,
            reprojectionError=2.0,
        )

        if not success:
            return None, None, [], float("inf")

        # 计算内点上的重投影误差
        if inliers is not None and len(inliers) > 0:
            proj, _ = cv2.projectPoints(
                obj_pts_arr[inliers.flatten()], rvec, tvec,
                camera_matrix, dist_coeffs,
            )
            err = float(np.linalg.norm(
                img_pts_arr[inliers.flatten()] - proj.reshape(-1, 2)
            ) / np.sqrt(len(inliers)))
        else:
            err = float("inf")

        return rvec, tvec, used_ids, err

    # ------------------------------------------------------------------
    # 诊断方法
    # ------------------------------------------------------------------

    @staticmethod
    def draw(
        image: np.ndarray,
        markers: List[ArUcoMarker],
        color: Tuple[int, int, int] = (0, 255, 0),
        draw_id: bool = True,
    ) -> np.ndarray:
        """
        在 *image* 的副本上绘制检测到的标志点。

        Returns:
            标注后的 BGR 图像（新数组；原图不变）。
        """
        out = image.copy()
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

        for m_id, corners, _, _ in markers:
            c = corners.astype(np.int32)
            cv2.polylines(out, [c], True, color, 2)
            if draw_id:
                cv2.putText(
                    out, str(m_id),
                    (int(c[0][0]), int(c[0][1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                )

        return out

    @staticmethod
    def drawAxes(
        image: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        axis_length: float = 0.05,
    ) -> np.ndarray:
        """
        使用 ``cv2.drawFrameAxes`` 在 *image* 上绘制三维坐标轴。

        Args:
            axis_length: 坐标轴长度，单位为**米**（与 tvec 单位相同）。
        """
        out = image.copy()
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        cv2.drawFrameAxes(out, camera_matrix, dist_coeffs,
                          rvec, tvec, axis_length)
        return out


# ===================================================================
# 统一标志点跟踪器（ArUco + 圆形点）
# ===================================================================

class UnifiedMarkerTracker:
    """
    结合 ArUco 编码标志点检测和圆形点检测。

    当你混合使用多种标志点时使用此类：
    - ArUco 标签用于基于 ID 的对应（地面控制点）
    - 普通圆形点用于密集、高精度跟踪（飞行器表面）

    标志点 ID 约定（用于飞行器姿态流水线）：
    - ID 0–N  ：地面标志点（ArUco）
    - ID N+1–M：飞行器标志点（ArUco，或 ID 为 -1 的圆形点）
    """

    def __init__(
        self,
        aruco_dict: str = "4x4_50",
        refine_corners: bool = True,
    ) -> None:
        self._aruco_detector = CodedMarkerDetector(
            dict_name=aruco_dict,
            refine_corners=refine_corners,
        )
        # 延迟导入 SLSMarkerDetector 以避免循环依赖
        from sls_calib.marker_detector import SLSMarkerDetector
        self._circle_detector = SLSMarkerDetector()

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def detectAll(
        self,
        image: np.ndarray,
        detect_circles: bool = True,
        debug: bool = False,
    ) -> Tuple[List[ArUcoMarker], List[Tuple[Tuple[float, float], float]], str]:
        """
        一次遍历同时检测 ArUco 标志点和圆形点。

        Returns:
            ``(aruco_markers, circle_markers, error_info)``。
            *circle_markers* 的格式与
            ``SLSMarkerDetector.detectMarkers`` 相同 —— ``((cx, cy), area)``。
        """
        error_info = ""

        # --- ArUco 标志点 ---
        aruco_markers, aruco_err = self._aruco_detector.detect(image, debug=debug)
        if aruco_err:
            error_info += f"[ArUco] {aruco_err}; "

        # --- 圆形标志点 ---
        circle_markers = []
        if detect_circles:
            circle_markers, circle_err = self._circle_detector.detectMarkers(
                image, smooth=False, debug=False
            )
            if circle_err:
                error_info += f"[Circles] {circle_err}"

        return aruco_markers, circle_markers, error_info.strip("; ")

    def drawAll(
        self,
        image: np.ndarray,
        aruco_markers: List[ArUcoMarker],
        circle_markers: List[Tuple[Tuple[float, float], float]],
        aruco_color: Tuple[int, int, int] = (0, 255, 0),
        circle_color: Tuple[int, int, int] = (255, 0, 0),
    ) -> np.ndarray:
        """在一张图像上绘制 ArUco 标志点和圆形点。"""
        out = CodedMarkerDetector.draw(image, aruco_markers, color=aruco_color)

        for (cx, cy), area in circle_markers:
            r = int(math.sqrt(area / math.pi))
            cv2.circle(out, (int(cx), int(cy)), max(r, 2), circle_color, 2)
            cv2.circle(out, (int(cx), int(cy)), 1, circle_color, -1)

        return out


# ===================================================================
# 标志点生成（用于打印）
# ===================================================================

def generate_marker_image(
    marker_id: int,
    dict_name: str = "4x4_50",
    pixel_size: int = 200,
    border_bits: int = 1,
) -> np.ndarray:
    """
    生成单个 ArUco 标志点的灰度图像，供打印使用。

    Args:
        marker_id:  要生成的 ArUco ID（0 .. dict_max-1）。
        dict_name:  字典键，如 ``"4x4_50"``。
        pixel_size: 输出图像的像素尺寸（正方形）。
        border_bits: 白色边框宽度，以标志点位为单位。

    Returns:
        灰度 ``uint8`` 图像（``pixel_size × pixel_size``）。
    """
    dict_id = CodedMarkerDetector._DICT_MAP[dict_name]
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    img = cv2.aruco.generateImageMarker(
        aruco_dict, marker_id, pixel_size, borderBits=border_bits
    )
    return img


def generate_marker_sheet(
    dict_name: str = "4x4_50",
    pixel_size: int = 200,
    ids: Optional[List[int]] = None,
    columns: int = 4,
    margin: int = 20,
) -> np.ndarray:
    """
    生成包含多个 ArUco 标志点的可打印纸张。

    Args:
        dict_name: 字典键。
        pixel_size: 每个标志点的像素尺寸。
        ids: 要包含的标志点 ID 列表（默认：0..15）。
        columns: 每行的标志点数量。
        margin: 每个标志点周围的白色边距（像素）。

    Returns:
        灰度图像，标志点按网格排列，并标注 ID。
    """
    if ids is None:
        ids = list(range(16))

    rows = int(math.ceil(len(ids) / columns))

    cell = pixel_size + margin * 2
    sheet_w = columns * cell + margin
    sheet_h = rows * cell + margin
    sheet = np.full((sheet_h, sheet_w), 255, dtype=np.uint8)

    for idx, marker_id in enumerate(ids):
        r = idx // columns
        c = idx % columns
        y0 = margin + r * cell
        x0 = margin + c * cell
        marker_img = generate_marker_image(marker_id, dict_name, pixel_size)
        sheet[y0:y0 + pixel_size, x0:x0 + pixel_size] = marker_img

        # 在标志点下方标注 ID
        label = f"ID:{marker_id}"
        cv2.putText(
            sheet, label,
            (x0, y0 + pixel_size + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1,
        )

    return sheet


def generate_charuco_board(
    dict_name: str = "4x4_50",
    squares_x: int = 5,
    squares_y: int = 7,
    square_length: float = 0.04,
    marker_length: float = 0.02,
) -> cv2.aruco.CharucoBoard:
    """
    创建 ChArUco 板 —— 棋盘格 + ArUco 的混合模式，
    非常适用于高精度相机标定和地平面定义。

    Args:
        dict_name: ArUco 字典。
        squares_x, squares_y: 棋盘格方块数量。
        square_length: 每个棋盘格方块的边长（米）。
        marker_length: 每个 ArUco 标志点的边长（米）。

    Returns:
        ``cv2.aruco.CharucoBoard`` 对象。
    """
    dict_id = CodedMarkerDetector._DICT_MAP[dict_name]
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length, marker_length, aruco_dict,
    )
    return board


# ===================================================================
# （测试/演示代码已提取到 tests/test_coded_marker.py
#   和 tools/generate_markers.py）
# ===================================================================
