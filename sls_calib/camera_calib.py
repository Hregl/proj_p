"""
相机标定 —— 源自 calib.cpp 的 Python 移植版本
==============================================
实现 SLS 标定流水线：
  检测标志点 → NDC 坐标转换 → 网格分配 → OpenCV 标定

类：
  CalibImage   — 对标 SLSImage：保存图像 + 检测到的圆 + 网格数据
  Calibrator   — 对标 SLSRenderer/SLSManager：统筹标定流程
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .marker_detector import SLSMarkerDetector, Marker

# ---------------------------------------------------------------------------
# 类型别名
# ---------------------------------------------------------------------------
Circle2D = Tuple[float, float]  # (x, y) 像素坐标
World3D = Tuple[float, float, float]  # (x, y, z) 世界坐标
DisplayCircle = Tuple[float, float, bool]  # (ndc_x, ndc_y, is_large)
CircleEntry = Tuple[Circle2D, World3D, bool, float]  # (2D, 3D, 是否有效, 半径)


# ---------------------------------------------------------------------------
# CalibImage  ——  单张标定靶图像
# ---------------------------------------------------------------------------

@dataclass
class CalibImage:
    """单张标定图像及其检测到的圆和网格分配。"""

    name: str
    image: np.ndarray
    selected: bool = True
    circles: List[Marker] = field(default_factory=list)  # ((x,y), area)
    display_circles: List[DisplayCircle] = field(default_factory=list)
    circle_array: List[CircleEntry] = field(default_factory=list)  # 长度 99

    # ------------------------------------------------------------------
    def create_display_circles(self) -> None:
        """
        将像素空间的圆心转换为归一化设备坐标
        （NDC, [-1, 1]），Y 轴翻转使其顶部 → 1，底部 → -1。
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
        核心标定流程（移植自 SLSImage::findCircleIndices）。

        1. 识别 5 个大基准圆               (面积 > threshold × max)
        2. 按几何位置排序：中心圆、远对圆、近对圆
        3. 计算单应性矩阵 → 目标网格
        4. 将每个检测到的圆映射到 11×9 的网格索引 → circle_array[99]

        Parameters
        ----------
        circle_interval : float
            相邻圆心之间的物理间距（世界单位）。
        large_circle_threshold : float
            最大圆面积的分数阈值，超过此值的圆被视为"大"基准圆
            （默认 0.75；原始 C++ 代码硬编码为 0.5）。
        target_coords : ndarray，形状为 (5, 2)，可选
            覆盖 5 个排序后基准圆的默认目标坐标。
            默认值匹配原始 C++ 布局。

        Returns
        -------
        error_info : str
            成功时为空；否则描述出错原因。
        """
        h, w = self.image.shape[:2]

        # -------- 1. 找到 5 个大圆 --------------------------------------
        if not self.circles:
            return f"图像 {self.name} 未检测到任何圆！\n"

        max_area = max(area for _, area in self.circles)
        if max_area <= 0:
            return f"图像 {self.name} 检测到的圆面积无效(max_area={max_area:.1f})\n"
        large_indices: List[int] = []
        large_circles: List[Circle2D] = []

        for i, ((cx, cy), area) in enumerate(self.circles):
            if area / max_area > large_circle_threshold:
                large_indices.append(i)
                large_circles.append((cx, cy))
                if (0 <= i < len(self.display_circles) and
                        len(self.display_circles) == len(self.circles)):
                    ndc_x, ndc_y, _ = self.display_circles[i]
                    self.display_circles[i] = (ndc_x, ndc_y, True)

        if len(large_circles) != 5:
            return (f"图像 {self.name} 未找到五大圆！"
                    f"(找到 {len(large_circles)} 个, 阈值={large_circle_threshold})\n")

        # -------- 2. 最小距离对 & 最大距离对 ----------------------------
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

        # -------- 3. 按几何位置对 5 个圆排序 ----------------------------
        # 中心 = 既不属于最小距离对也不属于最大距离对的那个
        used = {min_pair[0], min_pair[1], max_pair[0], max_pair[1]}
        center_candidates = [i for i in range(5) if i not in used]
        if not center_candidates:
            return (f"图像 {self.name} 五大圆几何排序失败: "
                    f"最小对{min_pair}与最大对{max_pair}覆盖了全部5个圆\n")
        center_circle = large_circles[center_candidates[0]]

        # Target grid coordinates for the 5 fiducial circles
        if target_coords is None:
            dst_pts = np.array([
                [600, 300],  # [0] center     → grid (5, 2)
                [300, 500],  # [1] far-left   → grid (2, 4)
                [900, 500],  # [2] far-right  → grid (8, 4)
                [600, 700],  # [3] near-left  → grid (5, 6)
                [700, 700],  # [4] near-right → grid (6, 6)
            ], dtype=np.float32)
        else:
            dst_pts = np.asarray(target_coords, dtype=np.float32)
            if dst_pts.shape != (5, 2):
                return f"target_coords shape must be (5,2), got {dst_pts.shape}\n"

        def cross_z(a: Circle2D, b: Circle2D) -> float:
            return a[0] * b[1] - a[1] * b[0]

        def try_assignment(pair_near: Tuple[int, int], dst_near: Tuple[int, int],
                           pair_far: Tuple[int, int], dst_far: Tuple[int, int],
                           swap_lr: bool = False
                           ) -> Tuple[List, int]:
            """Try one pair-to-target assignment. Returns (circle_array, n_assigned)."""
            sorted_lc: List[Circle2D] = [None] * 5
            sorted_lc[0] = center_circle

            # Base vector: center → midpoint of near pair
            mid_nx = 0.5 * (large_circles[pair_near[0]][0] + large_circles[pair_near[1]][0])
            mid_ny = 0.5 * (large_circles[pair_near[0]][1] + large_circles[pair_near[1]][1])
            base_vec = (mid_nx - center_circle[0], mid_ny - center_circle[1])

            def order_pair(pair, dst_indices):
                a = (large_circles[pair[0]][0] - center_circle[0],
                     large_circles[pair[0]][1] - center_circle[1])
                if cross_z(a, base_vec) < 0:
                    sorted_lc[dst_indices[0]], sorted_lc[dst_indices[1]] = \
                        large_circles[pair[0]], large_circles[pair[1]]
                else:
                    sorted_lc[dst_indices[0]], sorted_lc[dst_indices[1]] = \
                        large_circles[pair[1]], large_circles[pair[0]]

            order_pair(pair_near, dst_near)
            order_pair(pair_far, dst_far)

            if swap_lr:
                # Swap left-right within each pair
                sorted_lc[dst_near[0]], sorted_lc[dst_near[1]] = \
                    sorted_lc[dst_near[1]], sorted_lc[dst_near[0]]
                sorted_lc[dst_far[0]], sorted_lc[dst_far[1]] = \
                    sorted_lc[dst_far[1]], sorted_lc[dst_far[0]]

            # Compute homography and grid assignment
            src_pts = np.array([[x, y] for x, y in sorted_lc], dtype=np.float32)
            H, _ = cv2.findHomography(src_pts, dst_pts)
            if H is None:
                return [], 0

            all_src = np.array([[cx, cy] for cx, cy, *_ in
                                [(m[0][0], m[0][1]) for m in self.circles]],
                               dtype=np.float32).reshape(-1, 1, 2)
            all_dst = cv2.perspectiveTransform(all_src, H).reshape(-1, 2)

            circle_arr = [((0.0, 0.0), (0.0, 0.0, 0.0), False, 2.0)
                          for _ in range(99)]
            for i, ((tx, ty), (_, area)) in enumerate(zip(all_dst, self.circles)):
                gx = int(tx / 100.0 - 0.5)
                gy = int(ty / 100.0 - 0.5)
                if (0 <= gx <= 10 and 0 <= gy <= 8
                        and abs((gx + 1) * 100.0 - tx) < 10.0
                        and abs((gy + 1) * 100.0 - ty) < 10.0):
                    idx = gy * 11 + gx
                    px, py = self.circles[i][0]
                    circle_arr[idx] = (
                        (px, py),
                        (circle_interval * gx, circle_interval * gy, 0.0),
                        True,
                        math.sqrt(area),
                    )
            n_assigned = sum(1 for _, _, ok, _ in circle_arr if ok)
            return circle_arr, n_assigned

        # Try 4 combos: 2 near/far orientations × 2 left-right variants
        best_arr, best_n = [], 0
        for min_as_near in [True, False]:
            for swap_lr in [False, True]:
                if min_as_near:
                    arr, n = try_assignment(
                        min_pair, (3, 4), max_pair, (1, 2), swap_lr)
                else:
                    arr, n = try_assignment(
                        max_pair, (3, 4), min_pair, (1, 2), swap_lr)
                if n > best_n:
                    best_arr, best_n = arr, n

        if best_n < 10:
            return (f"图像 {self.name} 网格分配失败: "
                    f"仅{best_n}个有效点(需≥10)\n")
        self.circle_array = best_arr

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
# Calibrator  ——  统筹多图像标定
# ---------------------------------------------------------------------------

class Calibrator:
    """
    管理跨多张图像和多个相机的完整标定流水线。

    对标 calib.cpp 中的 SLSRenderer::extractCircles、
    SLSRenderer::calibrateCamera 和 SLSManager::calibrateScanner。
    """

    def __init__(self) -> None:
        self.detector = SLSMarkerDetector()

    # ------ 从所有（已选中的）图像中提取圆 -------------------------------
    def extract_circles(
        self,
        images: List[CalibImage],
        only_selected: bool = True,
        smooth: bool = True,
        debug: bool = False,
    ) -> str:
        """
        对每张（已选中的）图像执行标志点检测。

        调用后，每个 ``CalibImage.circles`` 即被填充。
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

    # ------ 单相机标定 ---------------------------------------------------
    def calibrate_camera(
        self,
        images: List[CalibImage],
        prefix: str,
        debug: bool = False,
    ) -> Tuple[str, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        使用所有 ``name`` 包含 *prefix* 的图像标定一台相机。

        返回 (报告字符串, 相机内参矩阵, 畸变系数)。
        """
        # 构建在所有匹配图像中均可见的圆的掩码
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

        # 收集 3D ↔ 2D 点对应关系
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

        # 构建报告
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

    # ------ 完整扫描仪标定 -----------------------------------------------
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
        顶层标定入口（移植自 SLSManager::calibrateScanner）。

        Parameters
        ----------
        images : CalibImage 列表
        camera_prefixes : 字符串列表
            用于匹配图像名称的前缀字符串（每台相机一个）。
        circle_interval : float
            圆心之间的物理间距。
        only_extrinsic : bool
            如果为 True，使用提供的内参矩阵而非重新计算。
        intrinsic_matrices : ndarray 列表，可选
            预先标定好的相机内参矩阵（only_extrinsic=True 时必需）。
        dist_coeffs_list : ndarray 列表，可选
            预先标定好的畸变系数。
        debug : bool
        """
        if not images:
            return "没有图像，无法标定！\n"

        # 将相机与图像匹配
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

        # 步骤 1：提取圆
        error = self.extract_circles(images, only_selected=False, smooth=True,
                                     debug=debug)
        if error:
            return error

        # 步骤 2：为每张图像分配网格
        for img in images:
            error += img.find_circle_indices(circle_interval, debug=debug,
                                             large_circle_threshold=large_circle_threshold)
        if error:
            return error

        # 步骤 3：标定每台相机
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

        # 汇总
        for i, pf in enumerate(matched_prefixes):
            print(f"\n=== 相机 '{pf}' ===")
            print(f"内参矩阵 K =\n{camera_matrices[i]}")
            print(f"畸变系数 D = {dist_coeffs[i].ravel()}")

        return ""


# ---------------------------------------------------------------------------
# 测试入口
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# （测试/演示代码已提取到 tests/test_camera_calib.py
#   和 tools/run_calibration.py）
# ---------------------------------------------------------------------------
