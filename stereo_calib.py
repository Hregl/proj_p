"""
Stereo Camera Calibration — dual-camera calibration and rectification.

Computes the rigid transform (R, T) between two cameras using
simultaneous observations of a calibration target, then produces
rectification maps for epipolar-aligned stereo pairs.

Supports multiple calibration-pattern types:
  - Chessboard       (cv2.findChessboardCorners)
  - Circular grid    (SLSMarkerDetector + grid assignment)
  - ChArUco board    (CodedMarkerDetector + cv2.aruco)

Typical workflow
----------------
  1. Calibrate each camera individually  →  K_left, dist_left,
                                            K_right, dist_right
  2. Load stereo image pairs             →  List[(imgL, imgR), …]
  3. Detect pattern in all pairs         →  object_points,
                                            left_image_points,
                                            right_image_points
  4. stereoCalibrate                     →  R, T, E, F
  5. stereoRectify                       →  R1, R2, P1, P2, Q
  6. initUndistortRectifyMap             →  remap maps
  7. remap                               →  rectified stereo pairs

References
----------
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

from marker_detector import SLSMarkerDetector
from camera_calib import CalibImage, Calibrator

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------


@dataclass
class StereoParams:
    """Results of stereo calibration (everything needed for rectification)."""

    K_left: np.ndarray        # 3×3  left camera matrix
    dist_left: np.ndarray     #      left distortion coefficients
    K_right: np.ndarray       # 3×3  right camera matrix
    dist_right: np.ndarray    #      right distortion coefficients

    R: np.ndarray             # 3×3  rotation: right cam → left cam
    T: np.ndarray             # 3×1  translation: right cam in left frame
    E: np.ndarray             # 3×3  essential matrix
    F: np.ndarray             # 3×3  fundamental matrix

    R1: np.ndarray            # 3×3  left rectification rotation
    R2: np.ndarray            # 3×3  right rectification rotation
    P1: np.ndarray            # 3×4  left rectified projection
    P2: np.ndarray            # 3×4  right rectified projection
    Q: np.ndarray             # 4×4  disparity-to-depth mapping

    image_size: Tuple[int, int]  # (width, height)

    rms_error: float = 0.0    # stereo calibration RMS reprojection error


# ===================================================================
# StereoCalibrator
# ===================================================================


class StereoCalibrator:
    """
    Stereo camera calibration and rectification.

    Handles the full pipeline:
      intrinsic calibration (optional) → stereo calibration → rectification.

    Pattern types
    -------------
    ``"chessboard"``
        Standard black-and-white chessboard.  The default and easiest
        to use.  *pattern_size* is ``(cols, rows)`` of **inner**
        corners and *square_size* is the side length in metres.

    ``"circles"``
        SLS-style circular-dot target (the same 11×9 grid used by
        ``Calibrator``).  Requires the dot detector and grid logic.

    ``"charuco"``
        ChArUco board (chessboard + ArUco).  Most robust — the
        ArUco markers provide automatic ID assignment even under
        partial occlusion.  *pattern_size* = ``(squares_x, squares_y)``.
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
            pattern_type: ``"chessboard"`` | ``"circles"`` | ``"charuco"``.
            pattern_size: ``(cols, rows)`` of inner corners or squares.
            square_size: Physical side length of one square (metres).
            aruco_dict_name: ArUco dictionary (for ChArUco).
            marker_size: Physical ArUco marker side length (metres).
        """
        if pattern_type not in (self._PATTERN_CHESS,
                                self._PATTERN_CIRCLES,
                                self._PATTERN_CHARUCO):
            raise ValueError(
                f"Unknown pattern_type '{pattern_type}'. "
                f"Use 'chessboard', 'circles', or 'charuco'."
            )

        self.pattern_type = pattern_type
        self.pattern_size = pattern_size
        self.square_size = square_size
        self.marker_size = marker_size
        self.aruco_dict_name = aruco_dict_name

        # Internal detectors (lazy-init for circles/charuco)
        self._circle_detector: Optional[SLSMarkerDetector] = None
        self._calibrator: Optional[Calibrator] = None
        self._charuco_board: Optional[cv2.aruco.CharucoBoard] = None
        self._aruco_detector: Optional[cv2.aruco.ArucoDetector] = None

        # Cached 3D object points for the pattern
        self._obj_points_cache: Optional[np.ndarray] = None

        # Calibration results
        self._K_left: Optional[np.ndarray] = None
        self._dist_left: Optional[np.ndarray] = None
        self._K_right: Optional[np.ndarray] = None
        self._dist_right: Optional[np.ndarray] = None
        self._stereo_params: Optional[StereoParams] = None

        # Rectification maps
        self._map_left_x: Optional[np.ndarray] = None
        self._map_left_y: Optional[np.ndarray] = None
        self._map_right_x: Optional[np.ndarray] = None
        self._map_right_y: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # 3D object points for the calibration pattern
    # ------------------------------------------------------------------

    def _get_object_points(self) -> np.ndarray:
        """Build (N, 3) array of pattern points in the board's local frame."""
        if self._obj_points_cache is not None:
            return self._obj_points_cache

        if self.pattern_type == self._PATTERN_CHESS:
            cols, rows = self.pattern_size
            pts = np.zeros((rows * cols, 3), dtype=np.float32)
            pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
            pts *= self.square_size

        elif self.pattern_type == self._PATTERN_CIRCLES:
            # 11×9 grid, 5 large fiducial circles
            # This is the SLS calibration board layout
            pts_list = []
            for row in range(9):
                for col in range(11):
                    # Large circles at specific grid positions
                    large_positions = {(5, 2), (2, 4), (8, 4), (5, 6), (6, 6)}
                    pts_list.append([col, row, 0.0])
            pts = np.array(pts_list, dtype=np.float32)
            pts[:, :2] *= self.square_size

        elif self.pattern_type == self._PATTERN_CHARUCO:
            sx, sy = self.pattern_size
            board = self._get_charuco_board()
            # Charuco board object points are the chessboard corners
            pts = np.array([
                [c * self.square_size, r * self.square_size, 0.0]
                for r in range(sy - 1) for c in range(sx - 1)
            ], dtype=np.float32)

        else:
            raise RuntimeError(f"Unknown pattern: {self.pattern_type}")

        self._obj_points_cache = pts
        return pts

    # ------------------------------------------------------------------
    # Pattern detection
    # ------------------------------------------------------------------

    def detect_pattern(
        self,
        image: np.ndarray,
        debug: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        """
        Detect calibration-pattern points in a single image.

        Returns:
            ``(corners, object_points, error)``.
            *corners* is ``(N, 1, 2)`` float array of 2D points
            (OpenCV convention), or ``None`` on failure.
            *object_points* is ``(N, 1, 3)`` of corresponding 3D coords.
        """
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.pattern_type == self._PATTERN_CHESS:
            return self._detect_chessboard(gray, debug)

        elif self.pattern_type == self._PATTERN_CIRCLES:
            return self._detect_circles(image, debug)  # needs BGR for detector

        elif self.pattern_type == self._PATTERN_CHARUCO:
            return self._detect_charuco(gray, debug)

        return None, None, f"Unknown pattern: {self.pattern_type}"

    def _detect_chessboard(
        self, gray: np.ndarray, debug: bool
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        cols, rows = self.pattern_size
        found, corners = cv2.findChessboardCornersSB(
            gray, (cols, rows),
            flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY,
        )
        if not found:
            return None, None, "Chessboard not found"

        # Sub-pixel refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

        obj_pts = self._get_object_points()

        return corners, obj_pts.reshape(-1, 1, 3), ""

    def _detect_circles(
        self, image: np.ndarray, debug: bool
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        # Lazy-init circle detector and calibrator
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
            return None, None, "No circles detected"

        calib_img.circles = markers
        calib_img.create_display_circles()
        err2 = calib_img.find_circle_indices(self.square_size, debug=debug)
        if err2:
            return None, None, err2

        # Extract valid 2D-3D correspondences from circle_array
        img_pts = []
        obj_pts = []
        for (px, py), (wx, wy, wz), valid, _ in calib_img.circle_array:
            if valid:
                img_pts.append([px, py])
                obj_pts.append([wx, wy, wz])

        if not img_pts:
            return None, None, "No valid grid assignments"

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
            return None, None, "Too few ArUco markers for ChArUco"

        n_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board
        )
        if charuco_corners is None or n_corners < 4:
            return None, None, "ChArUco corner interpolation failed"

        obj_pts, img_pts = cv2.aruco.getBoardObjectAndImagePoints(
            board, charuco_corners, charuco_ids
        )
        return img_pts, obj_pts, ""

    # ------------------------------------------------------------------
    # Intrinsic calibration (single camera)
    # ------------------------------------------------------------------

    def calibrate_intrinsics(
        self,
        images: List[np.ndarray],
        image_size: Optional[Tuple[int, int]] = None,
        debug: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Calibrate ONE camera using multiple views of the pattern.

        Args:
            images: List of images showing the same calibration pattern.
            image_size: ``(width, height)`` — auto-detected if None.

        Returns:
            ``(K, dist_coeffs, rms_error)`` or ``(None, None, inf)``.
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
            print(f"  Only {len(obj_pts_all)} valid images (need >= 3).")
            return None, None, float("inf")

        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_pts_all, img_pts_all, image_size,
            None, None,
            flags=cv2.CALIB_FIX_K3,
        )

        return K, dist.ravel(), ret

    # ------------------------------------------------------------------
    # Stereo calibration
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
        Calibrate stereo rig from pre-detected corner coordinates.

        This is the core calibration routine — it bypasses pattern
        detection and works directly with 2D↔3D correspondences.
        Use this when you already have corner positions (e.g. from
        synthetic data or from an external detector).

        Args:
            left_corners:  List of ``(N,1,2)`` arrays of 2D corner positions.
            right_corners: List of ``(N,1,2)`` arrays.
            object_points: ``(N,3)`` array of 3D board points.
            K_left, dist_left, K_right, dist_right: Intrinsics.
            image_size: ``(width, height)``.
            fix_intrinsics: Keep intrinsics fixed (recommended).

        Returns:
            ``StereoParams`` or ``None`` on failure.
        """
        n = len(left_corners)
        if n < 5:
            print(f"Need >= 5 pairs (got {n}).")
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

        # Rectification
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
        Calibrate the stereo rig: compute R, T, E, F between cameras.

        Detects the calibration pattern in each image pair, then
        calls ``calibrate_stereo_from_corners``.

        Returns:
            ``StereoParams`` or ``None`` on failure.
        """
        if len(left_images) != len(right_images):
            raise ValueError(
                f"Mismatched image counts: "
                f"{len(left_images)} left vs {len(right_images)} right"
            )
        if len(left_images) < 5:
            print(f"Need >= 5 stereo pairs (got {len(left_images)}).")
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
                    print(f"  Pair {idx}: skipped "
                          f"(L={'OK' if cornersL is not None else errL}, "
                          f"R={'OK' if cornersR is not None else errR})")
                continue

            obj_pts_all.append(objL.reshape(-1, 3))
            img_pts_left_all.append(cornersL.reshape(-1, 2))
            img_pts_right_all.append(cornersR.reshape(-1, 2))

        n_valid = len(obj_pts_all)
        if n_valid < 5:
            print(f"Only {n_valid} valid pairs (need >= 5).")
            return None

        if debug:
            print(f"  {n_valid}/{len(left_images)} pairs valid "
                  f"({(time.perf_counter() - t0) * 1000:.0f} ms)")

        return self.calibrate_stereo_from_corners(
            [c.reshape(-1, 1, 2) for c in img_pts_left_all],
            [c.reshape(-1, 1, 2) for c in img_pts_right_all],
            obj_pts_all[0],
            K_left, dist_left, K_right, dist_right,
            image_size, fix_intrinsics, debug,
        )

    # ------------------------------------------------------------------
    # Rectification
    # ------------------------------------------------------------------

    def _build_rectification_maps(self) -> None:
        """Pre-compute remap look-up tables for fast rectification."""
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
        Rectify a stereo pair.

        Returns ``(rectified_left, rectified_right)``, both aligned
        so that corresponding epipolar lines are horizontal.
        """
        if self._map_left_x is None:
            raise RuntimeError("Run calibrate_stereo() first.")

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
        Compute a disparity map from a stereo pair using SGBM.

        Args:
            left_image, right_image: The stereo pair.
            rectify_first: Apply rectification before matching.
            num_disparities: Must be divisible by 16.
            block_size: Must be odd.

        Returns:
            Disparity map (float32), scaled by 16.0 (divide by 16 to
            get pixel disparity).  Use ``stereo_params.Q`` to convert
            to depth.
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
        Convert disparity map to depth (metres) using the Q matrix.

        Returns depth map (float32) in metres, 0 where invalid.
        """
        if self._stereo_params is None:
            raise RuntimeError("Run calibrate_stereo() first.")

        points_3d = cv2.reprojectImageTo3D(
            disparity, self._stereo_params.Q, handleMissingValues=True
        )
        depth = points_3d[:, :, 2]  # Z channel
        return depth.astype(np.float32)

    # ------------------------------------------------------------------
    # Lazy-init helpers
    # ------------------------------------------------------------------

    def _get_charuco_board(self) -> cv2.aruco.CharucoBoard:
        if self._charuco_board is None:
            from coded_marker import CodedMarkerDetector
            dict_id = CodedMarkerDetector._DICT_MAP[self.aruco_dict_name]
            aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
            sx, sy = self.pattern_size
            self._charuco_board = cv2.aruco.CharucoBoard(
                (sx, sy), self.square_size, self.marker_size, aruco_dict,
            )
        return self._charuco_board

    def _get_aruco_detector(self) -> cv2.aruco.ArucoDetector:
        if self._aruco_detector is None:
            from coded_marker import CodedMarkerDetector
            dict_id = CodedMarkerDetector._DICT_MAP[self.aruco_dict_name]
            aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
            params = cv2.aruco.DetectorParameters()
            self._aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        return self._aruco_detector

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stereo_params(self) -> Optional[StereoParams]:
        """Results of the last ``calibrate_stereo()`` call."""
        return self._stereo_params

    @property
    def baseline(self) -> Optional[float]:
        """Stereo baseline in **metres** (norm of T)."""
        if self._stereo_params is None:
            return None
        return float(np.linalg.norm(self._stereo_params.T))

    @property
    def is_calibrated(self) -> bool:
        """Whether stereo calibration has been performed."""
        return self._stereo_params is not None

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable summary of stereo calibration."""
        if self._stereo_params is None:
            return "StereoCalibrator: not calibrated yet."

        sp = self._stereo_params
        lines = [
            "Stereo Calibration Summary",
            f"  Pattern:    {self.pattern_type} {self.pattern_size}",
            f"  Image size: {sp.image_size[0]}×{sp.image_size[1]}",
            f"  RMS error:  {sp.rms_error:.4f} px",
            f"  Baseline:   {self.baseline * 1000:.2f} mm",
            "",
            "  Left camera:",
            f"    K = [{sp.K_left[0,0]:.1f}, {sp.K_left[1,1]:.1f}] "
            f"cx,cy=({sp.K_left[0,2]:.1f}, {sp.K_left[1,2]:.1f})",
            f"    dist = {np.array2string(sp.dist_left, precision=4, suppress_small=True)}",
            "",
            "  Right camera:",
            f"    K = [{sp.K_right[0,0]:.1f}, {sp.K_right[1,1]:.1f}] "
            f"cx,cy=({sp.K_right[0,2]:.1f}, {sp.K_right[1,2]:.1f})",
            f"    dist = {np.array2string(sp.dist_right, precision=4, suppress_small=True)}",
            "",
            "  Stereo transform (right in left frame):",
            f"    R = {np.array2string(sp.R, precision=4, suppress_small=True)}",
            f"    T = {np.array2string(sp.T.ravel(), precision=4, suppress_small=True)}  (m)",
        ]
        return "\n".join(lines)


# ===================================================================
# Convenience: single-call stereo calibration (both intrinsics + extrinsics)
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
    One-shot stereo calibration: intrinsics → extrinsics → rectification.

    Convenience wrapper that runs intrinsic calibration for each camera
    followed by stereo calibration, all in one call.

    Args:
        left_images, right_images: Paired images of the calibration
            target.  Must be in sync (left_images[i] ↔ right_images[i]).
        pattern_type: ``"chessboard"`` | ``"circles"`` | ``"charuco"``.
        pattern_size: ``(cols, rows)`` of inner corners.
        square_size: Physical square side length in **metres**.
        debug: Print detailed progress.

    Returns:
        Calibrated ``StereoCalibrator``, or ``None`` on failure.
    """
    if len(left_images) != len(right_images):
        raise ValueError(
            f"Image count mismatch: {len(left_images)} vs {len(right_images)}"
        )

    calib = StereoCalibrator(
        pattern_type=pattern_type,
        pattern_size=pattern_size,
        square_size=square_size,
    )

    # --- Intrinsic calibration ---------------------------------------
    if debug:
        print("--- Left camera intrinsics ---")
    K_left, dist_left, rms_left = calib.calibrate_intrinsics(
        left_images, debug=debug
    )
    if K_left is None:
        print("Left camera intrinsic calibration failed.")
        return None
    if debug:
        print(f"  RMS = {rms_left:.4f} px")

    if debug:
        print("--- Right camera intrinsics ---")
    K_right, dist_right, rms_right = calib.calibrate_intrinsics(
        right_images, debug=debug
    )
    if K_right is None:
        print("Right camera intrinsic calibration failed.")
        return None
    if debug:
        print(f"  RMS = {rms_right:.4f} px")

    # --- Stereo calibration ------------------------------------------
    if debug:
        print("--- Stereo calibration ---")
    sp = calib.calibrate_stereo(
        left_images, right_images,
        K_left, dist_left,
        K_right, dist_right,
        debug=debug,
    )
    if sp is None:
        print("Stereo calibration failed.")
        return None

    if debug:
        print()
        print(calib.summary())

    return calib


# ===================================================================
# Synthetic-data test
# ===================================================================


def _generate_synthetic_stereo_data(
    n_pairs: int = 10,
    image_size: Tuple[int, int] = (1280, 720),
    pattern_size: Tuple[int, int] = (9, 6),
    square_size: float = 0.025,
    noise_px: float = 0.3,
) -> Tuple[
    List[np.ndarray],       # left images
    List[np.ndarray],       # right images
    List[np.ndarray],       # projected corners (left)
    List[np.ndarray],       # projected corners (right)
    np.ndarray,             # object points (N×3)
    np.ndarray,             # K_left
    np.ndarray,             # dist_left
    np.ndarray,             # K_right
    np.ndarray,             # dist_right
    np.ndarray,             # GT R (right in left frame)
    np.ndarray,             # GT T
]:
    """
    Generate synthetic stereo pairs of a chessboard target.

    Returns both rendered images (for visualisation) AND the projected
    corner coordinates (for direct injection into calibration, bypassing
    chessboard detection).  This is the standard approach for testing
    calibration math.
    """
    rng = np.random.default_rng(42)
    w, h = image_size
    cols, rows = pattern_size

    # Realistic camera intrinsics
    fx = 800.0
    K = np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)

    K_left = K.copy()
    dist_left = dist.copy()
    K_right = K.copy()
    dist_right = dist.copy()

    # Stereo baseline: right camera offset in +X
    baseline = 0.12  # 12 cm
    R_gt = np.eye(3)
    T_gt = np.array([baseline, 0.0, 0.0]).reshape(3, 1)

    # 3D chessboard corners (in board local frame, z=0)
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

        # Board at ~0.3–0.6 m, centred, moderate tilt
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

        # Project to left camera
        projL, _ = cv2.projectPoints(
            obj_pts, rvec_board, tvec_board, K_left, dist_left
        )
        # Right camera: X_right_cam = R_gt @ X_left_cam + T_gt
        #                = R_gt @ (R_board @ X_obj + tvec) + T_gt
        R_right = R_gt @ R_board
        rvec_right = cv2.Rodrigues(R_right)[0]
        tvec_right = (R_gt @ tvec_board.ravel() + T_gt.ravel()).astype(np.float64)
        projR, _ = cv2.projectPoints(
            obj_pts, rvec_right, tvec_right.reshape(3, 1),
            K_right, dist_right,
        )

        # Check in-bounds (with margin)
        projL_2d = projL.reshape(-1, 2)
        projR_2d = projR.reshape(-1, 2)
        margin = 40
        if not (
            np.all((projL_2d > margin) & (projL_2d < np.array([w - margin, h - margin])))
            and np.all((projR_2d > margin) & (projR_2d < np.array([w - margin, h - margin])))
        ):
            continue

        # Add noise to corner coords and draw
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
# Main
# ===================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Stereo Calibration — Synthetic Data Test")
    print("=" * 60)

    # Generate synthetic data with ground truth
    print("\n[1] Generating synthetic stereo pairs …")
    (
        left_imgs, right_imgs,
        left_corners, right_corners,
        obj_pts,
        K_left, dist_left, K_right, dist_right,
        R_gt, T_gt,
    ) = _generate_synthetic_stereo_data(n_pairs=15, noise_px=0.3)
    n = len(left_imgs)
    print(f"  Generated {n} valid stereo pairs "
          f"({left_imgs[0].shape[1]}x{left_imgs[0].shape[0]})")

    for i, (imgL, imgR) in enumerate(zip(left_imgs, right_imgs)):
        cv2.imwrite(f"stereo_left_{i}.png", imgL)
        cv2.imwrite(f"stereo_right_{i}.png", imgR)

    # Stereo calibration (bypasses pattern detection, injects corners)
    print("\n[2] Running stereo calibration (known corners) …")
    calib = StereoCalibrator(pattern_type="chessboard",
                             pattern_size=(9, 6),
                             square_size=0.025)

    sp = calib.calibrate_stereo_from_corners(
        left_corners, right_corners, obj_pts,
        K_left, dist_left,
        K_right, dist_right,
        image_size=(1280, 720),
        fix_intrinsics=True,
        debug=True,
    )

    if sp is None:
        print("Calibration failed.")
        exit(1)

    # Accuracy vs ground truth
    print("\n[3] Accuracy vs ground truth:")

    R_err_mat = sp.R @ R_gt.T
    angle_err = math.acos(
        np.clip((np.trace(R_err_mat) - 1) / 2, -1.0, 1.0)
    )
    print(f"  Rotation error:      {np.rad2deg(angle_err):.5f} deg")

    t_gt_norm = np.linalg.norm(T_gt)
    t_est_norm = np.linalg.norm(sp.T)
    baseline_err_pct = abs(t_est_norm - t_gt_norm) / t_gt_norm * 100
    print(f"  Baseline:  GT = {t_gt_norm*1000:.1f} mm  "
          f"Est = {t_est_norm*1000:.1f} mm  "
          f"({baseline_err_pct:.3f}% error)")

    t_gt_unit = T_gt.ravel() / t_gt_norm
    t_est_unit = sp.T.ravel() / t_est_norm
    dir_err = math.acos(
        np.clip(abs(np.dot(t_gt_unit, t_est_unit)), -1.0, 1.0)
    )
    print(f"  Direction error:     {np.rad2deg(dir_err):.5f} deg"
          f"{' (sign convention)' if np.dot(t_gt_unit, t_est_unit) < 0 else ''}")
    print(f"  Reprojection RMS:    {sp.rms_error:.5f} px")

    # Stereo rectification visual check
    print("\n[4] Rectification visual check …")
    rectL, rectR = calib.rectify(left_imgs[0], right_imgs[0])
    cv2.imwrite("stereo_rectified_L.png", rectL)
    cv2.imwrite("stereo_rectified_R.png", rectR)

    comparison = np.hstack([rectL, rectR])
    for y in range(0, comparison.shape[0], 60):
        cv2.line(comparison, (0, y), (comparison.shape[1], y), (0, 255, 0), 1)
    cv2.imwrite("stereo_rectified_comparison.png", comparison)
    print("  Wrote stereo_rectified_comparison.png")

    # Disparity
    print("\n[5] Disparity map …")
    disparity = calib.compute_disparity(left_imgs[0], right_imgs[0])
    disp_viz = np.clip(disparity / max(disparity.max(), 1) * 255, 0, 255).astype(np.uint8)
    cv2.imwrite("stereo_disparity.png", disp_viz)
    print("  Wrote stereo_disparity.png")

    print()
    print(calib.summary())
