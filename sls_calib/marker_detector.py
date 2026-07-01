"""
标志点检测器 —— 源自 markerdetector.cpp 中 SLSMarkerDetector 的 Python 移植版本

通过轮廓分析与亚像素精化，在图像中寻找圆形标志点。
"""

import math
import time
from typing import List, Tuple

import cv2
import numpy as np

# 类型别名：一个标志点表示为 ((cx, cy), area)
Marker = Tuple[Tuple[float, float], float]


class SLSMarkerDetector:
    """通过轮廓拟合与亚像素精化在图像中检测圆形标志点。"""

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def detectMarkers(
        self,
        image: np.ndarray,
        smooth: bool = False,
        debug: bool = False,
    ) -> Tuple[List[Marker], str]:
        """
        在图像中检测圆形标志点。

        Args:
            image:  输入图像（BGR 或灰度 numpy 数组）。
            smooth: 处理前是否应用 GaussianBlur。
            debug:  是否输出调试图像（mask.png, contour.png, circle.png）。

        Returns:
            (markers, error_info)，其中 *markers* 是一个
            ``((cx, cy), area)`` 元组的列表，*error_info* 在成功时为
            空字符串，否则为错误描述。
        """
        markers: List[Marker] = []

        if image is None or image.size == 0:
            return markers, "标志点图像为空！\n"

        t0 = time.perf_counter()
        error_info = ""

        try:
            if image.ndim == 3 and image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            self._findCircles(markers, gray, smooth, debug)
        except cv2.error as exc:
            error_info = str(exc)

        elapsed = (time.perf_counter() - t0) * 1000.0
        print(f"findCircles: {elapsed:.3f}ms")

        return markers, error_info

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _timed_print(name: str, t0: float) -> float:
        """打印自 *t0* 起经过的毫秒数，并返回新的时间戳。"""
        now = time.perf_counter()
        print(f"{name}: {(now - t0) * 1000.0:.3f}ms")
        return now

    def _findCircles(
        self,
        markers: List[Marker],
        image: np.ndarray,
        smooth: bool,
        debug: bool,
    ) -> None:
        height, width = image.shape[:2]
        min_size = width / 200.0

        t = time.perf_counter()

        # --- 平滑处理 ---------------------------------------------------
        if smooth:
            smoothed = cv2.GaussianBlur(image, (3, 3), 0.0)
        else:
            smoothed = image

        # --- 阈值分割 ---------------------------------------------------
        _, mask = cv2.threshold(smoothed, 40, 255, cv2.THRESH_BINARY)
        t = self._timed_print("threshold", t)

        # --- 边缘检测 + 轮廓提取 ---------------------------------------
        edge = cv2.Canny(smoothed, 50, 150, 3)

        # OpenCV 4.x 返回 (contours, hierarchy)
        contours, _ = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        t = self._timed_print("contour", t)

        # --- Sobel 梯度 -------------------------------------------------
        sobel_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
        sobel_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)
        t = self._timed_print("sobel", t)

        # --- 亚像素精化 -------------------------------------------------
        circles: list = []          # cv2.fitEllipse 结果
        subpixel_contours: list = []  # 精化后的轮廓，用于调试

        t = time.perf_counter()

        for contour in contours:
            if len(contour) < 5:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w <= min_size and h <= min_size:
                continue

            area = abs(cv2.contourArea(contour))
            length = cv2.arcLength(contour, True)
            # 圆形度检查：area/length > length / (4*pi) * 0.7
            if area / length <= length / (4.0 * math.pi) * 0.7:
                continue

            ellipse = cv2.fitEllipse(contour)
            cx = int(ellipse[0][0] + 0.5)
            cy = int(ellipse[0][1] + 0.5)

            if not (0 <= cx < width and 0 <= cy < height):
                continue
            if mask[cy, cx] != 255:
                continue

            # --- 亚像素点精化 --------------------------------------------
            inside = True
            subpixel_pts: List[List[float]] = []

            for pt in contour:
                px, py = int(pt[0][0]), int(pt[0][1])

                if px < 2 or py < 2 or px >= width - 2 or py >= height - 2:
                    inside = False
                    break

                delta = self._subpixelFit((px, py), image, sobel_x, sobel_y)
                if delta[0] ** 2 + delta[1] ** 2 < 1.0:
                    subpixel_pts.append([px + delta[0], py + delta[1]])

            if inside and len(subpixel_pts) >= 5:
                pts_arr = np.array(subpixel_pts, dtype=np.float32)
                subpixel_contours.append(pts_arr.reshape(-1, 1, 2).astype(np.int32))
                ellipse = cv2.fitEllipse(pts_arr)
                circles.append(ellipse)

        t = self._timed_print("subpixel", t)

        # --- 去重：移除重叠的圆 -----------------------------------------
        clean_circles: list = []
        processed = [False] * len(circles)

        for i in range(len(circles)):
            if processed[i]:
                continue

            # size.area() = 旋转矩形的宽 × 高
            min_area = circles[i][1][0] * circles[i][1][1]
            min_area_circle = circles[i]
            cxi, cyi = circles[i][0]

            for j in range(len(circles)):
                if i == j or processed[j]:
                    continue

                cxj, cyj = circles[j][0]
                dist = math.hypot(cxi - cxj, cyi - cyj)
                if dist < 5.0:
                    area_j = circles[j][1][0] * circles[j][1][1]
                    if area_j < min_area:
                        min_area = area_j
                        min_area_circle = circles[j]
                    processed[j] = True

            clean_circles.append(min_area_circle)
            markers.append(((min_area_circle[0][0], min_area_circle[0][1]), min_area))
            processed[i] = True

        # --- 调试输出 ---------------------------------------------------
        if debug:
            contour_img = np.zeros((height, width, 3), dtype=np.uint8)
            circle_img = np.zeros((height, width), dtype=np.uint8)

            rng = np.random.default_rng()
            for i, sp_cnt in enumerate(subpixel_contours):
                color = (
                    int(rng.integers(0, 256)),
                    int(rng.integers(0, 256)),
                    int(rng.integers(0, 256)),
                )
                cv2.drawContours(contour_img, subpixel_contours, i, color, 1)

            for c in clean_circles:
                cv2.ellipse(circle_img, c, 255, 1)

            import os as _os
            _os.makedirs("output", exist_ok=True)
            cv2.imwrite("output/mask.png", mask)
            cv2.imwrite("output/contour.png", contour_img)
            cv2.imwrite("output/circle.png", circle_img)

    # ------------------------------------------------------------------
    # 亚像素精化（两个重载版本）
    # ------------------------------------------------------------------

    def _subpixelFit(
        self,
        point: Tuple[int, int],
        image: np.ndarray,
        sobel_x: np.ndarray,
        sobel_y: np.ndarray,
    ) -> Tuple[float, float]:
        """
        使用预计算的 Sobel 梯度进行一维亚像素精化。

        在穿过 *point* 的一条 5 像素线段上（根据局部梯度方向选择水平或
        垂直方向）采样，并计算梯度幅值的加权质心。
        """
        px, py = point
        height, width = image.shape[:2]

        diff_x = float(sobel_x[py, px])
        diff_y = float(sobel_y[py, px])

        sum_wx = 0.0
        sum_wy = 0.0
        sum_x = 0.0
        sum_y = 0.0

        def accumulate(x: int, y: int) -> None:
            nonlocal sum_x, sum_y, sum_wx, sum_wy
            sx = float(sobel_x[y, x])
            sy = float(sobel_y[y, x])
            sum_x += sx * (x - px)
            sum_y += sy * (y - py)
            sum_wx += sx
            sum_wy += sy

        angle = math.degrees(math.atan2(diff_y, diff_x))

        if (45.0 < angle < 135.0) or (-135.0 < angle < -45.0):
            # 梯度主要为垂直方向 → 沿垂直线采样
            y0 = max(0, py - 2)
            y1 = min(height, py + 3)
            for y in range(y0, y1):
                accumulate(px, y)
        else:
            # 梯度主要为水平方向 → 沿水平线采样
            x0 = max(0, px - 2)
            x1 = min(width, px + 3)
            for x in range(x0, x1):
                accumulate(x, py)

        if sum_wx == 0.0 or sum_wy == 0.0:
            return (0.0, 0.0)

        return (sum_x / sum_wx, sum_y / sum_wy)

    # ------------------------------------------------------------------
    # 备选方案：二维多项式曲面拟合（当前未被调用）
    # ------------------------------------------------------------------

    @staticmethod
    def _subpixelFitPoly(
        point: Tuple[int, int],
        image: np.ndarray,
        diff_x: float,
        diff_y: float,
    ) -> Tuple[float, float]:
        """
        二维亚像素精化：在 5×5 邻域上用三次多项式曲面拟合，沿梯度方向
        求解过零点。

        注意：保留此方法以与 C++ API 保持一致，但在默认检测流水线中未被
        使用。
        """
        px, py = point

        # 构建 25×10 的设计矩阵和 25×1 的观测灰度值
        mat_b = np.empty((25, 10), dtype=np.float64)
        mat_y = np.empty((25, 1), dtype=np.float64)

        row = 0
        for ky in range(2, -3, -1):       # k =  2, 1, 0, -1, -2
            for jx in range(-2, 3):       # j = -2, -1, 0,  1,  2
                ty = float(ky)
                tx = float(jx)

                mat_b[row, 0] = 1.0
                mat_b[row, 1] = tx
                mat_b[row, 2] = ty
                mat_b[row, 3] = tx * tx
                mat_b[row, 4] = tx * ty
                mat_b[row, 5] = ty * ty
                mat_b[row, 6] = tx * tx * tx
                mat_b[row, 7] = tx * tx * ty
                mat_b[row, 8] = ty * ty * tx
                mat_b[row, 9] = ty * ty * ty
                mat_y[row, 0] = float(image[py + ky, px + jx])
                row += 1

        # 最小二乘求解： coeffs = (BᵗB)⁻¹ Bᵗ y
        coeffs, _, _, _ = np.linalg.lstsq(mat_b, mat_y, rcond=None)
        k = coeffs.flatten()  # 10 个三次曲面系数

        angle = math.atan2(diff_y, diff_x)
        c = math.cos(angle)
        s = math.sin(angle)

        # p = -1/3 * (二次项) / (三次项)，沿梯度方向求值
        # （推导参见原始 C++ 代码）。
        numerator = (
            k[3] * c * c
            + k[4] * c * s
            + k[5] * s * s
        )
        denominator = (
            k[6] * c * c * c
            + k[7] * c * c * s
            + k[8] * c * s * s
            + k[9] * s * s * s
        )
        if denominator == 0.0:
            return (0.0, 0.0)

        p = (numerator / denominator) * (-1.0 / 3.0)
        return (float(p * c), float(p * s))


# ----------------------------------------------------------------------
# （测试/演示代码已提取到 tests/test_marker_detector.py）
# ----------------------------------------------------------------------
