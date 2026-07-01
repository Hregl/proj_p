"""
Camera Calibration — Python port of calib.cpp
==============================================
Implements the SLS calibration pipeline:
  detect markers → NDC conversion → grid assignment → OpenCV calibration

Classes:
  CalibImage   — mirrors SLSImage: holds image + detected circles + grid data
  Calibrator   — mirrors SLSRenderer/SLSManager: orchestrates calibration
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from marker_detector import SLSMarkerDetector, Marker

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Circle2D = Tuple[float, float]  # (x, y) in pixels
World3D = Tuple[float, float, float]  # (x, y, z) in world units
DisplayCircle = Tuple[float, float, bool]  # (ndc_x, ndc_y, is_large)
CircleEntry = Tuple[Circle2D, World3D, bool, float]  # (2D, 3D, valid, radius)


# ---------------------------------------------------------------------------
# CalibImage  —  a single calibration-target image
# ---------------------------------------------------------------------------

@dataclass
class CalibImage:
    """One calibration image with its detected circles and grid assignments."""

    name: str
    image: np.ndarray
    selected: bool = True
    circles: List[Marker] = field(default_factory=list)  # ((x,y), area)
    display_circles: List[DisplayCircle] = field(default_factory=list)
    circle_array: List[CircleEntry] = field(default_factory=list)  # length 99

    # ------------------------------------------------------------------
    def create_display_circles(self) -> None:
        """
        Convert pixel-space circle centres to Normalised Device Coordinates
        (NDC, [-1, 1]) with Y flipped so top → 1, bottom → -1.
        """
        if not self.circles:
            return
        h, w = self.image.shape[:2]
        inv_w = 1.0 / w
        inv_h = 1.0 / h

        self.display_circles = []
        for (cx, cy), area in self.circles:
            ndc_x = (cx + 0.5) * inv_w * 2.0 - 1.0
            ndc_y = 1.0 - (cy + 0.5) * inv_h * 2.0
            ndc_r = math.sqrt(area) * inv_w
            self.display_circles.append((ndc_x, ndc_y, False))

    # ------------------------------------------------------------------
    def find_circle_indices(
        self, circle_interval: float, debug: bool = False,
        large_circle_threshold: float = 0.75,
        target_coords: Optional[np.ndarray] = None,
    ) -> str:
        """
        Core calibration routine (port of SLSImage::findCircleIndices).

        1. Identify the 5 large fiducial circles          (area > threshold × max)
        2. Sort them geometrically: centre, far-pair, near-pair
        3. Compute homography → target grid
        4. Map every detected circle to an 11×9 grid index → circle_array[99]

        Parameters
        ----------
        circle_interval : float
            Physical spacing between adjacent circle centres (world units).
        large_circle_threshold : float
            Fraction of max circle area above which a circle is considered
            a "large" fiducial (default 0.75; original C++ hardcoded 0.5).
        target_coords : ndarray of shape (5, 2), optional
            Override the default target coordinates for the 5 sorted fiducial
            circles. Default matches the original C++ layout.

        Returns
        -------
        error_info : str
            Empty on success; describes what went wrong otherwise.
        """
        h, w = self.image.shape[:2]

        # -------- 1. find the 5 large circles --------------------------------
        if not self.circles:
            return f"图像 {self.name} 未检测到任何圆！\n"

        max_area = max(area for _, area in self.circles)
        large_indices: List[int] = []
        large_circles: List[Circle2D] = []

        for i, ((cx, cy), area) in enumerate(self.circles):
            if area / max_area > large_circle_threshold:
                large_indices.append(i)
                large_circles.append((cx, cy))
                if len(self.display_circles) == len(self.circles):
                    ndc_x, ndc_y, _ = self.display_circles[i]
                    self.display_circles[i] = (ndc_x, ndc_y, True)

        if len(large_circles) != 5:
            return f"图像 {self.name} 未找到五大圆！(找到 {len(large_circles)} 个)\n"

        # -------- 2. min-distance pair & max-distance pair --------------------
        min_dist = float("inf")
        max_dist = 0.0
        min_pair = (0, 1)
        max_pair = (0, 1)

        for i in range(5):
            for j in range(i + 1, 5):
                d = math.hypot(large_circles[i][0] - large_circles[j][0],
                               large_circles[i][1] - large_circles[j][1])
                if d < min_dist:
                    min_dist = d
                    min_pair = (i, j)
                if d > max_dist:
                    max_dist = d
                    max_pair = (i, j)

        # -------- 3. sort the 5 circles geometrically -------------------------
        sorted_lc: List[Circle2D] = [None] * 5  # type: ignore[assignment]

        # index [0] = the one that is neither min-pair nor max-pair
        used = {min_pair[0], min_pair[1], max_pair[0], max_pair[1]}
        for i in range(5):
            if i not in used:
                sorted_lc[0] = large_circles[i]
                break

        base = (
            0.5 * (large_circles[min_pair[0]][0] + large_circles[min_pair[1]][0]),
            0.5 * (large_circles[min_pair[0]][1] + large_circles[min_pair[1]][1]),
        )
        base_vec = (base[0] - sorted_lc[0][0], base[1] - sorted_lc[0][1])

        def cross_z(a: Circle2D, b: Circle2D) -> float:
            """Z-component of cross product (a - origin) × b."""
            return a[0] * b[1] - a[1] * b[0]

        # near pair → indices [3], [4]  (right-handed order)
        a = (large_circles[min_pair[0]][0] - sorted_lc[0][0],
             large_circles[min_pair[0]][1] - sorted_lc[0][1])
        if cross_z(a, base_vec) < 0:
            sorted_lc[3], sorted_lc[4] = large_circles[min_pair[0]], large_circles[min_pair[1]]
        else:
            sorted_lc[3], sorted_lc[4] = large_circles[min_pair[1]], large_circles[min_pair[0]]

        # far pair → indices [1], [2]  (right-handed order)
        b = (large_circles[max_pair[0]][0] - sorted_lc[0][0],
             large_circles[max_pair[0]][1] - sorted_lc[0][1])
        if cross_z(b, base_vec) < 0:
            sorted_lc[1], sorted_lc[2] = large_circles[max_pair[0]], large_circles[max_pair[1]]
        else:
            sorted_lc[1], sorted_lc[2] = large_circles[max_pair[1]], large_circles[max_pair[0]]

        if debug:
            for i, (sx, sy) in enumerate(sorted_lc):
                cv2.circle(self.image,
                           (int(sx + 0.5), int(sy + 0.5)),
                           10, (0, 0, 0), i + 1)

        # -------- 4. homography from 5 large circles → target grid ------------
        src_pts = np.array([[x, y] for x, y in sorted_lc], dtype=np.float32)
        if target_coords is None:
            dst_pts = np.array([
                [600, 300],  # [0] centre   → grid (5, 2)
                [300, 500],  # [1] far-left  → grid (2, 4)
                [900, 500],  # [2] far-right → grid (8, 4)
                [600, 700],  # [3] near-left → grid (5, 6)
                [700, 700],  # [4] near-right→ grid (6, 6)
            ], dtype=np.float32)
        else:
            dst_pts = np.asarray(target_coords, dtype=np.float32)
            if dst_pts.shape != (5, 2):
                return f"target_coords 形状必须为 (5, 2)，实际为 {dst_pts.shape}\n"

        H, _ = cv2.findHomography(src_pts, dst_pts)

        # -------- 5. map all circles to the 11×9 grid -------------------------
        self.circle_array = [((0.0, 0.0), (0.0, 0.0, 0.0), False, 2.0)
                             for _ in range(99)]

        all_src = np.array([[cx, cy] for cx, cy, *_ in
                            [(m[0][0], m[0][1]) for m in self.circles]],
                           dtype=np.float32).reshape(-1, 1, 2)
        all_dst = cv2.perspectiveTransform(all_src, H).reshape(-1, 2)

        for i, ((tx, ty), (_, area)) in enumerate(zip(all_dst, self.circles)):
            gx = int(tx / 100.0 - 0.5)
            gy = int(ty / 100.0 - 0.5)
            if (0 <= gx <= 10 and 0 <= gy <= 8
                    and abs((gx + 1) * 100.0 - tx) < 10.0
                    and abs((gy + 1) * 100.0 - ty) < 10.0):
                idx = gy * 11 + gx
                px, py = self.circles[i][0]
                self.circle_array[idx] = (
                    (px, py),
                    (circle_interval * gx, circle_interval * gy, 0.0),
                    True,
                    math.sqrt(area),
                )

        if debug:
            for gy in range(9):
                line = ""
                for gx in range(11):
                    _, _, valid, _ = self.circle_array[gy * 11 + gx]
                    line += "1 " if valid else "0 "
                print(line)
            print()

        return ""


# ---------------------------------------------------------------------------
# Calibrator  —  orchestrates multi-image calibration
# ---------------------------------------------------------------------------

class Calibrator:
    """
    Manages the full calibration pipeline across multiple images and cameras.

    Mirrors SLSRenderer::extractCircles, SLSRenderer::calibrateCamera, and
    SLSManager::calibrateScanner from calib.cpp.
    """

    def __init__(self) -> None:
        self.detector = SLSMarkerDetector()

    # ------ extract circles from all (selected) images --------------------
    def extract_circles(
        self,
        images: List[CalibImage],
        only_selected: bool = True,
        smooth: bool = True,
        debug: bool = False,
    ) -> str:
        """
        Run marker detection on every (selected) image.

        After this call each ``CalibImage.circles`` is populated.
        """
        error_info = ""
        count = 0
        for img in images:
            if only_selected and not img.selected:
                continue
            markers, err = self.detector.detectMarkers(
                img.image, smooth=smooth, debug=debug
            )
            img.circles = markers
            error_info += err
            if debug:
                print(count)
                count += 1
                for (cx, cy), _ in markers:
                    print(f"{cx:f} {cy:f}")
            img.create_display_circles()
        return error_info

    # ------ single-camera calibration -------------------------------------
    def calibrate_camera(
        self,
        images: List[CalibImage],
        prefix: str,
        debug: bool = False,
    ) -> Tuple[str, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Calibrate ONE camera using all images whose ``name`` contains
        *prefix*.

        Returns (report_string, camera_matrix, dist_coeffs).
        """
        # Build mask of circles visible in EVERY matching image
        if not images:
            return ("没有图像！\n", None, None)

        common_circles = [True] * 99
        found_any = False
        for img in images:
            if prefix not in img.name:
                continue
            if not img.circle_array:
                return (f"图像 {img.name} 尚未调用 find_circle_indices()！\n", None, None)
            found_any = True
            for i, (_, _, valid, _) in enumerate(img.circle_array):
                if not valid:
                    common_circles[i] = False

        if not found_any:
            return (f"未找到相机 '{prefix}' 的图像！\n", None, None)

        if debug:
            for i in range(99):
                print("1 " if common_circles[i] else "0 ", end="")
                if (i + 1) % 11 == 0:
                    print()
            print()

        # Collect 3D ↔ 2D point correspondences
        obj_pts_all: List[List[np.ndarray]] = []
        img_pts_all: List[List[np.ndarray]] = []
        radii: List[float] = []

        for img in images:
            if prefix not in img.name:
                continue
            obj_pts_one: List[np.ndarray] = []
            img_pts_one: List[np.ndarray] = []
            for i, common in enumerate(common_circles):
                if not common:
                    continue
                (px, py), (wx, wy, wz), _, r = img.circle_array[i]
                obj_pts_one.append(np.array([wx, wy, wz], dtype=np.float32))
                img_pts_one.append(np.array([px, py], dtype=np.float32))
                radii.append(r)
            if obj_pts_one:
                obj_pts_all.append(obj_pts_one)
                img_pts_all.append(img_pts_one)

        if not obj_pts_all:
            return ("没有公共圆点可用于标定！\n", None, None)

        img_size = images[0].image.shape[:2][::-1]  # (w, h)

        try:
            ret, mtx, dist, rvecs, tvecs, std_int, std_ext, per_view = (
                cv2.calibrateCamera(
                    obj_pts_all, img_pts_all, img_size,
                    None, None,
                    flags=cv2.CALIB_FIX_K3,
                )
            )
        except cv2.error as exc:
            return (str(exc), None, None)

        # Build report
        lines = [
            f"相机 '{prefix}' 内参矩阵：",
            str(mtx),
            f"相机 '{prefix}' 畸变系数：",
            str(dist),
            f"相机 '{prefix}' 内参标准差：",
            str(std_int),
            f"相机 '{prefix}' 外参标准差：",
            str(std_ext),
            f"相机 '{prefix}' 重投影误差：{ret}",
            str(per_view),
        ]
        return ("\n".join(lines), mtx, dist)

    # ------ full scanner calibration --------------------------------------
    def calibrate_scanner(
        self,
        images: List[CalibImage],
        camera_prefixes: List[str],
        circle_interval: float,
        only_extrinsic: bool = False,
        intrinsic_matrices: Optional[List[np.ndarray]] = None,
        dist_coeffs_list: Optional[List[np.ndarray]] = None,
        large_circle_threshold: float = 0.75,
        debug: bool = False,
    ) -> str:
        """
        Top-level calibration entry point (port of SLSManager::calibrateScanner).

        Parameters
        ----------
        images : list of CalibImage
        camera_prefixes : list of str
            Prefix strings to match against image names (one per camera).
        circle_interval : float
            Physical spacing between circle centres.
        only_extrinsic : bool
            If True, use supplied intrinsic matrices instead of computing them.
        intrinsic_matrices : list of ndarray, optional
            Pre-calibrated camera matrices (required when only_extrinsic=True).
        dist_coeffs_list : list of ndarray, optional
            Pre-calibrated distortion coefficients.
        debug : bool
        """
        if not images:
            return "没有图像，无法标定！\n"

        # Match cameras to images
        matched_prefixes: List[str] = []
        for pf in camera_prefixes:
            for img in images:
                if pf in img.name:
                    matched_prefixes.append(pf)
                    break

        if not matched_prefixes:
            return "没有图像对应的相机，无法标定！\n"
        if len(matched_prefixes) > 2:
            return "图像对应的相机数量超过2！\n"

        # Step 1: extract circles
        error = self.extract_circles(images, only_selected=False, smooth=True,
                                     debug=debug)
        if error:
            return error

        # Step 2: grid assignment for each image
        for img in images:
            error += img.find_circle_indices(circle_interval, debug=debug,
                                             large_circle_threshold=large_circle_threshold)
        if error:
            return error

        # Step 3: calibrate each camera
        if only_extrinsic:
            if intrinsic_matrices is None or dist_coeffs_list is None:
                return "仅外参模式下必须提供内参矩阵和畸变系数！\n"
            camera_matrices = list(intrinsic_matrices)
            dist_coeffs = list(dist_coeffs_list)
        else:
            camera_matrices = [None] * len(matched_prefixes)
            dist_coeffs = [None] * len(matched_prefixes)
            for i, pf in enumerate(matched_prefixes):
                report, K, D = self.calibrate_camera(images, pf, debug=debug)
                print(report)
                if K is not None:
                    camera_matrices[i] = K
                    dist_coeffs[i] = D
                else:
                    error += report
        if error:
            return error

        # Summary
        for i, pf in enumerate(matched_prefixes):
            print(f"\n=== 相机 '{pf}' ===")
            print(f"内参矩阵 K =\n{camera_matrices[i]}")
            print(f"畸变系数 D = {dist_coeffs[i].ravel()}")

        return ""


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

def main() -> None:
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "p1.png"
    circle_interval = float(sys.argv[2]) if len(sys.argv) > 2 else 35.0

    print(f"加载图像: {path}")
    img = cv2.imread(path)
    if img is None:
        print(f"错误: 无法读取图像 '{path}'")
        sys.exit(1)
    print(f"图像尺寸: {img.shape[1]} x {img.shape[0]}")

    # Build a single-image list for testing
    calib_img = CalibImage(name="test", image=img.copy(), selected=True)
    images = [calib_img]

    calibrator = Calibrator()

    # --- detect markers ---
    print("\n--- 检测标志点 ---")
    err = calibrator.extract_circles(images, only_selected=False,
                                     smooth=True, debug=True)
    if err:
        print(f"错误: {err}")
        sys.exit(1)
    print(f"检测到 {len(calib_img.circles)} 个圆")

    # --- print display circles (first 5) ---
    print("\n--- 显示坐标 (前5个) ---")
    for i, dc in enumerate(calib_img.display_circles[:5]):
        print(f"  [{i}] ndc=({dc[0]:.4f}, {dc[1]:.4f})  large={dc[2]}")

    # --- grid assignment ---
    print(f"\n--- 网格分配 (间距={circle_interval}) ---")
    err = calib_img.find_circle_indices(circle_interval, debug=True,
                                         large_circle_threshold=0.78)
    if err:
        print(f"错误: {err}")
        sys.exit(1)

    # --- print populated grid ---
    print("\n--- 圆点阵列 (11×9 grid) ---")
    valid_count = 0
    for gy in range(9):
        line = ""
        for gx in range(11):
            _, _, ok, _ = calib_img.circle_array[gy * 11 + gx]
            if ok:
                valid_count += 1
                line += "● "
            else:
                line += "· "
        print(line)
    print(f"\n有效圆点: {valid_count}/99")

    # --- show which are the 5 large circles in grid coords ---
    print("\n--- 五大圆网格坐标 ---")
    for i, ((cx, cy), area) in enumerate(calib_img.circles):
        if area / max(a for _, a in calib_img.circles) > 0.5:
            # Find grid index
            for gi, ((px, py), (wx, wy, _), ok, _) in enumerate(calib_img.circle_array):
                if ok and abs(px - cx) < 2 and abs(py - cy) < 2:
                    print(f"  大圆 ({cx:.1f}, {cy:.1f}) → grid({gi % 11}, {gi // 11})  "
                          f"world=({wx:.1f}, {wy:.1f})")
                    break

    print("\n完成。")


if __name__ == "__main__":
    main()
