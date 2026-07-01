"""
Marker Detector - Python port of SLSMarkerDetector from markerdetector.cpp

Finds circular markers in images using contour analysis with subpixel refinement.
"""

import math
import time
from typing import List, Tuple

import cv2
import numpy as np

# Type alias: a marker is ((cx, cy), area)
Marker = Tuple[Tuple[float, float], float]


class SLSMarkerDetector:
    """Detects circular markers in images via contour fitting and subpixel refinement."""

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detectMarkers(
        self,
        image: np.ndarray,
        smooth: bool = False,
        debug: bool = False,
    ) -> Tuple[List[Marker], str]:
        """
        Detect circular markers in an image.

        Args:
            image:  Input image (BGR or grayscale numpy array).
            smooth: Apply GaussianBlur before processing.
            debug:  Write debug images (mask.png, contour.png, circle.png).

        Returns:
            (markers, error_info) where *markers* is a list of
            ``((cx, cy), area)`` tuples and *error_info* is an empty
            string on success or an error description.
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
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _timed_print(name: str, t0: float) -> float:
        """Print elapsed milliseconds since *t0* and return a new timestamp."""
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

        # --- smooth ---------------------------------------------------
        if smooth:
            smoothed = cv2.GaussianBlur(image, (3, 3), 0.0)
        else:
            smoothed = image

        # --- threshold ------------------------------------------------
        _, mask = cv2.threshold(smoothed, 40, 255, cv2.THRESH_BINARY)
        t = self._timed_print("threshold", t)

        # --- edge + contours ------------------------------------------
        edge = cv2.Canny(smoothed, 50, 150, 3)

        # OpenCV 4.x returns (contours, hierarchy)
        contours, _ = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        t = self._timed_print("contour", t)

        # --- Sobel gradients ------------------------------------------
        sobel_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
        sobel_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)
        t = self._timed_print("sobel", t)

        # --- subpixel refinement --------------------------------------
        circles: list = []          # cv2.fitEllipse results
        subpixel_contours: list = []  # refined contours for debug

        t = time.perf_counter()

        for contour in contours:
            if len(contour) < 5:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w <= min_size and h <= min_size:
                continue

            area = abs(cv2.contourArea(contour))
            length = cv2.arcLength(contour, True)
            # Circularity check: area/length > length / (4*pi) * 0.7
            if area / length <= length / (4.0 * math.pi) * 0.7:
                continue

            ellipse = cv2.fitEllipse(contour)
            cx = int(ellipse[0][0] + 0.5)
            cy = int(ellipse[0][1] + 0.5)

            if not (0 <= cx < width and 0 <= cy < height):
                continue
            if mask[cy, cx] != 255:
                continue

            # --- subpixel point refinement ----------------------------
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

        # --- deduplicate overlapping circles --------------------------
        clean_circles: list = []
        processed = [False] * len(circles)

        for i in range(len(circles)):
            if processed[i]:
                continue

            # size.area() = width * height of the rotated rect
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

        # --- debug output ---------------------------------------------
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
    # Subpixel refinement (two overloads)
    # ------------------------------------------------------------------

    def _subpixelFit(
        self,
        point: Tuple[int, int],
        image: np.ndarray,
        sobel_x: np.ndarray,
        sobel_y: np.ndarray,
    ) -> Tuple[float, float]:
        """
        1-D subpixel refinement using pre-computed Sobel gradients.

        Samples a 5-pixel line through *point* — either horizontal or
        vertical depending on the local gradient direction — and computes
        a weighted centroid of the gradient magnitudes.
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
            # Gradient is mostly vertical → sample along vertical line
            y0 = max(0, py - 2)
            y1 = min(height, py + 3)
            for y in range(y0, y1):
                accumulate(px, y)
        else:
            # Gradient is mostly horizontal → sample along horizontal line
            x0 = max(0, px - 2)
            x1 = min(width, px + 3)
            for x in range(x0, x1):
                accumulate(x, py)

        if sum_wx == 0.0 or sum_wy == 0.0:
            return (0.0, 0.0)

        return (sum_x / sum_wx, sum_y / sum_wy)

    # ------------------------------------------------------------------
    # Alternative 2-D polynomial surface fit (not currently called)
    # ------------------------------------------------------------------

    @staticmethod
    def _subpixelFitPoly(
        point: Tuple[int, int],
        image: np.ndarray,
        diff_x: float,
        diff_y: float,
    ) -> Tuple[float, float]:
        """
        2-D subpixel refinement using a cubic polynomial surface fit
        over a 5×5 neighbourhood and solving for the zero-crossing along
        the gradient direction.

        Note: kept for parity with the C++ API but not used in the
        default detection pipeline.
        """
        px, py = point

        # Build 25×10 design matrix and 25×1 observed grey-values
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

        # Solve least-squares:  coeffs = (BᵗB)⁻¹ Bᵗ y
        coeffs, _, _, _ = np.linalg.lstsq(mat_b, mat_y, rcond=None)
        k = coeffs.flatten()  # 10 cubic-surface coefficients

        angle = math.atan2(diff_y, diff_x)
        c = math.cos(angle)
        s = math.sin(angle)

        # p = -1/3 * (quadratic terms) / (cubic terms) evaluated along
        # the gradient direction (see C++ original for derivation).
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
# (Test / demo code has been extracted to tests/test_marker_detector.py)
# ----------------------------------------------------------------------
