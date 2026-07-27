"""
Calibration board circle detector — connected-components approach.

Uses binary threshold + connected components on the mask to find
bright circular blobs on a dark board. More robust than Canny edges
for low-contrast / shallow-color calibration targets.

Flow:
  1. Grayscale → GaussianBlur(3,3)
  2. Fixed threshold(40) → binary mask (white = bright circles, black = bg)
  3. connectedComponentsWithStats → extract white blobs
  4. Filter by area, width, height, aspect ratio
  5. Center = component centroid

This is designed for SLS-type boards where circles are distinctly
brighter than the dark background, even if their edges are soft.
"""
import math, cv2, numpy as np
from typing import List, Tuple, Optional

Marker = Tuple[Tuple[float, float], float]  # ((cx, cy), area)


class BoardDetector:
    """Connected-component based circle detector for calibration boards."""

    def __init__(self,
                 threshold: int = 40,
                 min_area: float = 100,
                 max_area: float = 20000,
                 min_side: float = 8,
                 max_side: float = 220,
                 min_aspect: float = 0.35,
                 max_aspect: float = 2.85):
        self.threshold = threshold
        self.min_area = min_area
        self.max_area = max_area
        self.min_side = min_side
        self.max_side = max_side
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect

    def detect(self, image: np.ndarray,
               debug: bool = False,
               debug_dir: str = "output"
               ) -> Tuple[List[Marker], Optional[np.ndarray]]:
        """Alias for detect_and_filter()."""
        return self.detect_and_filter(image, debug=debug, debug_dir=debug_dir)

    def detect_and_filter(self, image: np.ndarray,
                          debug: bool = False,
                          debug_dir: str = "output"
                          ) -> Tuple[List[Marker], Optional[np.ndarray]]:
        """
        Detect bright circles on a dark calibration board.

        Args:
            image: BGR or grayscale image.
            debug: if True, save mask + detection visualization.
            debug_dir: output directory for debug images.

        Returns:
            (markers, debug_image) where markers is [((cx,cy), area), ...]
            sorted by area descending.
        """
        if image is None or image.size == 0:
            return [], None

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        height, width = gray.shape[:2]

        # --- Step 1: Gaussian blur -----------------------------------------
        smoothed = cv2.GaussianBlur(gray, (3, 3), 0.0)

        # --- Step 2: Fixed threshold → binary mask -------------------------
        _, mask = cv2.threshold(smoothed, self.threshold, 255,
                                cv2.THRESH_BINARY)

        # --- Step 3: Connected components ----------------------------------
        n_labels, labels, stats, centroids = \
            cv2.connectedComponentsWithStats(mask, connectivity=8)

        # --- Step 4: Filter components -------------------------------------
        markers: List[Marker] = []

        for i in range(1, n_labels):  # skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            left = stats[i, cv2.CC_STAT_LEFT]
            top = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            # Area filter
            if area < self.min_area or area > self.max_area:
                continue

            # Width / height bounds
            if w < self.min_side or w > self.max_side:
                continue
            if h < self.min_side or h > self.max_side:
                continue

            # Aspect ratio
            if h > 0:
                aspect = w / h
            else:
                continue
            if aspect < self.min_aspect or aspect > self.max_aspect:
                continue

            # Center = component centroid
            cx = centroids[i, 0]
            cy = centroids[i, 1]

            markers.append(((float(cx), float(cy)), float(area)))

        # --- Step 5: Deduplicate by centroid proximity ---------------------
        markers.sort(key=lambda x: -x[1])  # largest first
        clean: List[Marker] = []
        for m in markers:
            too_close = False
            for c in clean:
                if math.hypot(m[0][0] - c[0][0], m[0][1] - c[0][1]) < 5.0:
                    too_close = True
                    break
            if not too_close:
                clean.append(m)

        # --- Debug visualization -------------------------------------------
        dbg_img = None
        if debug:
            import os
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(f"{debug_dir}/board_mask.png", mask)

            dbg_img = image.copy() if image.ndim == 3 else \
                       cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            for (cx, cy), area in clean:
                # Green ellipse roughly matching the component
                radius = int(math.sqrt(area / math.pi))
                cv2.ellipse(dbg_img,
                           (int(cx), int(cy)),
                           (radius, radius), 0, 0, 360,
                           (0, 255, 0), 1)
                # Red cross at center
                s = 3
                cv2.line(dbg_img, (int(cx)-s, int(cy)), (int(cx)+s, int(cy)),
                        (0, 0, 255), 1)
                cv2.line(dbg_img, (int(cx), int(cy)-s), (int(cx), int(cy)+s),
                        (0, 0, 255), 1)

            cv2.imwrite(f"{debug_dir}/board_detected.png", dbg_img)

        return clean, dbg_img


# ====================================================================
# Unified entry point for all tools
# ====================================================================

class BoardDetectionResult:
    """Structured output from detect_and_assign_board()."""
    def __init__(self):
        self.point_ids: List[str] = []         # Board point IDs (B001, B002, ...)
        self.image_points: List[Tuple[float, float]] = []  # (u, v)
        self.object_points: List[Tuple[float, float, float]] = []  # (x, y, z)
        self.detected_count: int = 0
        self.assigned_count: int = 0
        self.assignment_rmse: float = 0.0
        self.orientation: str = ''
        self.rvec = None
        self.tvec = None
        self.success: bool = False
        self.inlier_ratio: float = 0.0

    def __repr__(self):
        return (f"BoardDetectionResult(assigned={self.assigned_count}/{self.detected_count}, "
                f"rmse={self.assignment_rmse:.3f}px, "
                f"orientation={self.orientation}, "
                f"success={self.success})")


def detect_and_assign_board(image: np.ndarray,
                            K: np.ndarray,
                            dist: np.ndarray,
                            circle_interval: float = 25.0,
                            debug: bool = False
                            ) -> BoardDetectionResult:
    """Unified board detection + grid assignment + PnP validation.

    This is the SINGLE entry point for calibration board processing.
    All tools (detect_board_points, estimate_board_pose, triangulation)
    should call this function rather than duplicating threshold logic.

    Args:
        image: BGR or grayscale image.
        K: Camera intrinsics (3x3).
        dist: Distortion coefficients.
        circle_interval: Circle spacing in mm (default 25 for SLS boards).

    Returns:
        BoardDetectionResult with point correspondences and PnP validation.
    """
    from sls_calib.camera_calib import CalibImage

    result = BoardDetectionResult()

    # Step 1: Detect circles via connected-components
    detector = BoardDetector()
    markers, _ = detector.detect_and_filter(image, debug=debug)
    result.detected_count = len(markers)

    if result.detected_count < 10:
        return result

    # Step 2: Grid assignment via auto-tune (same logic as BoardPoseEstimator)
    ci = CalibImage(name='tmp', image=image, selected=True)
    ci.circles = [((cx, cy), area) for (cx, cy), area in markers]
    ci.create_display_circles()

    best_score, best_arr = -1, None
    for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        ci2 = CalibImage(name='tmp2', image=image, selected=True)
        ci2.circles = ci.circles
        ci2.display_circles = ci.display_circles
        err = ci2.find_circle_indices(circle_interval, large_circle_threshold=t)
        if err != '':
            continue
        n = sum(1 for _, _, ok, _ in ci2.circle_array if ok)
        if n < 6:
            continue

        obj_t, img_t = [], []
        for (px, py), (wx, wy, wz), ok2, _ in ci2.circle_array:
            if ok2:
                obj_t.append([wx, wy, wz])
                img_t.append([px, py])
        if len(obj_t) < 6:
            continue

        obj_t_arr = np.array(obj_t, dtype=np.float64)
        img_t_arr = np.array(img_t, dtype=np.float64)

        ok_pnp, rv, tv = cv2.solvePnP(
            obj_t_arr, img_t_arr, K, dist, flags=cv2.SOLVEPNP_IPPE)
        if not ok_pnp:
            continue

        proj_t, _ = cv2.projectPoints(obj_t_arr, rv, tv, K, dist)
        rmse_t = float(np.sqrt(np.mean(
            np.linalg.norm(proj_t.reshape(-1, 2) - img_t_arr, axis=1) ** 2)))
        if rmse_t > 10.0:
            continue

        score = n - rmse_t * 5
        if score > best_score:
            best_score = score
            best_arr = ci2.circle_array
            result.rvec = rv
            result.tvec = tv
            result.assignment_rmse = rmse_t

    if best_arr is None:
        return result

    # Step 3: Extract results with correct point IDs.
    # The circle_array is 99 elements indexed by gy*11+gx.
    # Point ID B001 = grid(0,0), B002 = grid(1,0), ..., B{idx+1:03d}.
    total_cells = 99
    for idx, ((px, py), (wx, wy, wz), ok, _) in enumerate(best_arr):
        if ok:
            point_id = f'B{idx + 1:03d}'
            result.point_ids.append(point_id)
            result.image_points.append((px, py))
            result.object_points.append((wx, wy, wz))

    result.assigned_count = len(result.point_ids)

    # Quality assessment
    # Detect which orientation was chosen based on the circle_array pattern
    # (not just hardcoded 'normal')
    n_top_rows = sum(1 for i in range(11) if best_arr[i][2])  # gy=0 row
    n_bot_rows = sum(1 for i in range(88, 99) if best_arr[i][2])  # gy=8 row
    if n_bot_rows > n_top_rows:
        result.orientation = 'near_pair_bottom'
    else:
        result.orientation = 'near_pair_top'

    # Success criteria (experiment-grade):
    # - >= 80 out of 99 points assigned (board mostly visible)
    # - assignment RMSE <= 2.0 px (full-set IPPE RMSE, not RANSAC-inlier RMSE)
    result.success = (result.assigned_count >= 80 and
                      result.assignment_rmse <= 2.0)
    result.inlier_ratio = result.assigned_count / total_cells if total_cells > 0 else 0.0

    return result


# ====================================================================
# BoardPoseEstimator — legacy wrapper for tools that need the old API
# ====================================================================

class BoardPoseEstimator:
    """Detect calibration board and estimate camera pose via PnP.

    Provides backward-compatible API for tools that were previously
    coupled to the old BoardPoseEstimator in triangulate_aircraft_points.py.

    Note: In the new pipeline, the calibration board is ONLY used for
    camera intrinsics (Stage 1). Ground coordinate system (G) is defined
    by measured ground markers (GroundPoseEstimator).
    """

    def __init__(self, K: np.ndarray, dist: np.ndarray,
                 board_yaml: str = "configs/calibration_board_points.yaml",
                 circle_interval: float = 25.0,
                 large_thresh: float = 0.55,
                 use_blob: bool = True):
        import yaml as _yaml
        self.K = np.asarray(K, dtype=np.float64)
        self.dist = np.asarray(dist, dtype=np.float64).ravel()
        with open(board_yaml, encoding='utf-8') as f:
            board = _yaml.safe_load(f)
        self.pts3d = np.array(list(board['points'].values()), dtype=np.float64)
        self.pt_ids = list(board['points'].keys())
        self.interval = circle_interval
        self.thresh = large_thresh
        self.use_blob = use_blob

    def process_image(self, img: np.ndarray):
        """Detect board, assign grid, run PnP.

        Returns:
            G_R_C, G_t_C, rmse, rvec, tvec
        """
        import cv2, numpy as np
        from sls_calib.camera_calib import CalibImage

        ci = CalibImage(name='tmp', image=img, selected=True)

        if self.use_blob:
            bd = BoardDetector()
            markers, _ = bd.detect(img)
            ci.circles = [((cx, cy), area) for (cx, cy), area in markers]
            ci.create_display_circles()
        else:
            from sls_calib import Calibrator
            calib = Calibrator()
            calib.extract_circles([ci], only_selected=False,
                                  smooth=True, debug=False)
            ci.create_display_circles()

        # Auto-tune thresholds
        best_score, best_arr = -1, None
        thresholds = [self.thresh] if self.thresh else \
                     [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
        for t in thresholds:
            ci2 = CalibImage(name='tmp2', image=img, selected=True)
            ci2.circles = ci.circles
            ci2.display_circles = ci.display_circles
            err = ci2.find_circle_indices(self.interval, debug=False,
                                          large_circle_threshold=t)
            if err != '':
                continue
            n_assigned = sum(1 for _, _, ok, _ in ci2.circle_array if ok)
            if n_assigned < 6:
                continue
            obj_t, img_t = [], []
            for (px, py), (wx, wy, wz), ok2, _ in ci2.circle_array:
                if ok2:
                    obj_t.append([wx, wy, wz])
                    img_t.append([px, py])
            obj_t = np.array(obj_t, dtype=np.float64)
            img_t = np.array(img_t, dtype=np.float64)
            ok_pnp, rv, tv = cv2.solvePnP(
                obj_t, img_t, self.K, self.dist,
                flags=cv2.SOLVEPNP_IPPE)
            if not ok_pnp:
                continue
            proj_t, _ = cv2.projectPoints(obj_t, rv, tv, self.K, self.dist)
            errs_t = np.linalg.norm(proj_t.reshape(-1, 2) - img_t, axis=1)
            rmse_t = float(np.sqrt(np.mean(errs_t ** 2)))
            if rmse_t > 10.0:
                continue
            score = n_assigned - rmse_t * 5
            if score > best_score:
                best_score = score
                best_arr = ci2.circle_array

        if best_arr is None:
            return None, None, 999, None, None

        obj, img_pts = [], []
        for (px, py), (wx, wy, wz), ok, _ in best_arr:
            if ok:
                obj.append([wx, wy, wz])
                img_pts.append([px, py])

        obj_arr = np.array(obj, dtype=np.float64)
        img_arr = np.array(img_pts, dtype=np.float64)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_arr, img_arr, self.K, self.dist,
            flags=cv2.SOLVEPNP_IPPE, iterationsCount=200,
            reprojectionError=2.0, confidence=0.99)
        if not success:
            success, rvec, tvec = cv2.solvePnP(
                obj_arr, img_arr, self.K, self.dist,
                flags=cv2.SOLVEPNP_IPPE)
            inliers = None
        if not success:
            return None, None, 999, None, None

        C_R_G, _ = cv2.Rodrigues(rvec)
        C_t_G = tvec.ravel()
        G_R_C = np.linalg.inv(C_R_G)
        G_t_C = -G_R_C @ C_t_G

        if inliers is not None and len(inliers) > 0:
            inl_idx = inliers.ravel()
            mask = np.zeros(len(obj_arr), dtype=bool)
            mask[inl_idx] = True
            proj, _ = cv2.projectPoints(
                obj_arr[mask], rvec, tvec, self.K, self.dist)
            errs = np.linalg.norm(proj.reshape(-1, 2) - img_arr[mask], axis=1)
        else:
            proj, _ = cv2.projectPoints(obj_arr, rvec, tvec, self.K, self.dist)
            errs = np.linalg.norm(proj.reshape(-1, 2) - img_arr, axis=1)
        rmse = float(np.sqrt(np.mean(errs ** 2)))

        return G_R_C, G_t_C, rmse, rvec, tvec

