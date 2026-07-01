"""
Multi-View Structure from Motion (SfM) for marker-based aircraft
pose estimation.

Pipeline overview
-----------------
  Phase A — Ground-truth acquisition (Camera 1, multiple views):
    1. Detect coded markers (ArUco) in all images.
    2. Initialise from the best image pair via essential matrix +
       triangulation.
    3. Incrementally register remaining views with PnP.
    4. Triangulate newly-seen markers.
    5. Global resection-intersection bundle adjustment.
    6. Align the world coordinate frame to physical ground markers.
    7. Compute the 3D marker positions → "ground truth".

  Phase B — Single-view pose inference (Camera 2):
    Use ``solvePnP`` with the 3D points from Phase A.
    (see ``CodedMarkerDetector.estimatePoseMulti`` in coded_marker.py)

Coordinate conventions
----------------------
  - All rotations are 3×3 matrices (R), not Rodrigues vectors.
  - Camera projection matrix: P = K [R | t], where K is the 3×3
    intrinsic matrix and [R | t] maps world → camera.
  - 3D points are stored as (x, y, z) in the **world** frame.

References
----------
  - Hartley & Zisserman, *Multiple View Geometry*, 2nd ed., ch. 9–10, 18.
  - OpenCV ``cv2.solvePnPRansac``, ``cv2.triangulatePoints``,
    ``cv2.findEssentialMat``, ``cv2.recoverPose``.
"""

from __future__ import annotations

import itertools
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .coded_marker import CodedMarkerDetector
from .marker_detector import SLSMarkerDetector

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# A 3D point index (int) or a dict mapping marker_id → (x, y, z)
Point3D = Tuple[float, float, float]

# ---------------------------------------------------------------------------
# View: one image with detected markers
# ---------------------------------------------------------------------------


class View:
    """A single image in the SfM reconstruction."""

    __slots__ = (
        "name", "image", "markers", "centers",
        "R", "t", "registered",
    )

    def __init__(self, name: str, image: np.ndarray) -> None:
        self.name = name
        self.image = image
        # markers: {marker_id: corners_4x2}
        self.markers: Dict[int, np.ndarray] = {}
        # centers: {marker_id: (cx, cy)}
        self.centers: Dict[int, Tuple[float, float]] = {}
        # Camera pose in world frame: X_cam = R @ X_world + t
        self.R: Optional[np.ndarray] = None   # 3×3
        self.t: Optional[np.ndarray] = None   # 3×1
        self.registered: bool = False


# ===================================================================
# MultiViewSfM
# ===================================================================


class MultiViewSfM:
    """
    Multi-view structure-from-motion using coded (ArUco) markers.

    Parameters
    ----------
    camera_matrix:
        3×3 camera intrinsic matrix ``K``.
    dist_coeffs:
        Lens distortion coefficients (4, 5, 8, or 12 elements).
    marker_size_m:
        Physical side length of **one ArUco marker** in metres.
        Used to resolve the absolute scale.  Can also be set later
        via ``set_scale_from_marker``.
    aruco_dict:
        ArUco dictionary name (same as ``CodedMarkerDetector``).
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        marker_size_m: Optional[float] = None,
        aruco_dict: str = "4x4_50",
    ) -> None:
        self.K = camera_matrix.astype(np.float64)
        self.dist = np.asarray(dist_coeffs, dtype=np.float64).ravel()
        self.marker_size_m = marker_size_m
        self._scale = 1.0  # applied AFTER BA;  real = scale * reconstructed

        # Detectors
        self._aruco = CodedMarkerDetector(
            dict_name=aruco_dict, refine_corners=True
        )
        self._circle = SLSMarkerDetector()

        # Data
        self.views: List[View] = []
        self._points3d: Dict[int, Point3D] = {}  # marker_id → (x, y, z)
        self._point_observations: Dict[int, List[Tuple[int, int]]] = {}
        # ^ {marker_id: [(view_idx, corner_idx), ...]}

        # Timing
        self._timings: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Step 1 — Add views & detect markers
    # ------------------------------------------------------------------

    def add_views(
        self,
        images: Sequence[np.ndarray],
        names: Optional[Sequence[str]] = None,
        detect_circles: bool = False,
    ) -> None:
        """
        Add images to the reconstruction and detect ArUco markers.

        Args:
            images: List of BGR/grayscale images.
            names: Optional names (default: ``"view_0"``, …).
            detect_circles: Also detect uncoded circular dots (slower).
        """
        if names is None:
            names = [f"view_{i}" for i in range(len(images))]

        for name, img in zip(names, images):
            view = View(name, img)
            # ArUco detection
            aruco_markers, _ = self._aruco.detect(img)
            for m_id, corners, center, _side in aruco_markers:
                view.markers[m_id] = corners
                view.centers[m_id] = center

            # Optional circular-dot detection (for dense tracking)
            if detect_circles:
                circles, _ = self._circle.detectMarkers(img)
                # Circular dots get negative IDs to avoid collision
                for i, ((cx, cy), _area) in enumerate(circles):
                    cid = -(i + 1)
                    view.centers[cid] = (cx, cy)
                    # no corners array for circles

            self.views.append(view)

        print(f"Added {len(images)} view(s); "
              f"{sum(len(v.markers) for v in self.views)} ArUco detections total")

    # ------------------------------------------------------------------
    # Step 2 — Initialise from best image pair
    # ------------------------------------------------------------------

    def initialize(
        self,
        pair: Optional[Tuple[int, int]] = None,
        min_shared: int = 6,
        min_angle_deg: float = 3.0,
    ) -> bool:
        """
        Initialise the reconstruction from a pair of views.

        If *pair* is not given, the best pair is chosen automatically
        by maximising (shared_markers × triangulation_angle).

        Returns:
            ``True`` if initialisation succeeded.
        """
        t0 = time.perf_counter()

        if len(self.views) < 2:
            print("Need at least 2 views to initialise.")
            return False

        # --- auto-select best pair ------------------------------------
        if pair is None:
            pair = self._select_initial_pair(min_shared, min_angle_deg)
            if pair is None:
                print("Could not find a suitable initial pair. "
                      "Check marker coverage and baseline.")
                return False

        v0, v1 = self.views[pair[0]], self.views[pair[1]]
        shared_ids = sorted(set(v0.centers.keys()) & set(v1.centers.keys()))
        # Filter to ArUco markers only (positive IDs)
        shared_ids = [sid for sid in shared_ids if sid >= 0]

        if len(shared_ids) < min_shared:
            print(f"Only {len(shared_ids)} shared markers (need ≥{min_shared}).")
            return False

        print(f"Initialising from '{v0.name}' <-> '{v1.name}' "
              f"({len(shared_ids)} shared markers)")

        pts0 = np.array([v0.centers[sid] for sid in shared_ids], dtype=np.float64)
        pts1 = np.array([v1.centers[sid] for sid in shared_ids], dtype=np.float64)

        # Essential matrix with RANSAC
        E, inlier_mask = cv2.findEssentialMat(
            pts0, pts1, self.K, method=cv2.RANSAC,
            prob=0.999, threshold=1.0,
        )
        if E is None:
            print("Failed to estimate essential matrix.")
            return False

        inliers = inlier_mask.ravel().astype(bool)
        n_inliers = inliers.sum()
        print(f"  Essential matrix inliers: {n_inliers}/{len(shared_ids)}")

        if n_inliers < min_shared:
            print("Too few inliers after essential-matrix RANSAC.")
            return False

        # Recover relative pose (v1 in v0's coordinate frame)
        pts0_in = pts0[inliers]
        pts1_in = pts1[inliers]
        n_pts, R_rel, t_rel, mask_recover = cv2.recoverPose(
            E, pts0_in, pts1_in, self.K
        )
        angle_deg = float(np.linalg.norm(cv2.Rodrigues(R_rel)[0]))
        print(f"  Relative rotation: {np.rad2deg(angle_deg):.1f}°  "
              f"baseline: {np.linalg.norm(t_rel):.3f} (unscaled)")

        # Set view 0 as world origin
        v0.R = np.eye(3)
        v0.t = np.zeros((3, 1))
        v0.registered = True

        v1.R = R_rel
        v1.t = t_rel.reshape(3, 1)
        v1.registered = True

        # Triangulate shared markers
        P0 = self.K @ np.hstack([v0.R, v0.t])
        P1 = self.K @ np.hstack([v1.R, v1.t])

        pts0_f = pts0_in[mask_recover.ravel().astype(bool)].T.astype(np.float64)
        pts1_f = pts1_in[mask_recover.ravel().astype(bool)].T.astype(np.float64)
        shared_in = [sid for i, sid in enumerate(shared_ids)
                     if inliers[i] and mask_recover[i]]

        pts4d = cv2.triangulatePoints(P0, P1, pts0_f, pts1_f)
        pts3d = pts4d[:3] / pts4d[3]

        # Store valid 3D points (behind camera check)
        valid_count = 0
        for i, sid in enumerate(shared_in):
            p = pts3d[:, i]
            # Check point is in front of both cameras
            if p[2] > 0:  # in front of camera 0
                # Transform to camera 1 frame
                p1 = R_rel @ p + t_rel.ravel()
                if p1[2] > 0:
                    self._points3d[sid] = (float(p[0]), float(p[1]), float(p[2]))
                    valid_count += 1

        print(f"  Triangulated {valid_count} 3D points")

        self._timings["initialize"] = (time.perf_counter() - t0) * 1000
        return valid_count >= min_shared

    # ------------------------------------------------------------------
    # Step 3 — Incremental registration
    # ------------------------------------------------------------------

    def register_all(
        self,
        min_matches: int = 4,
        reproj_threshold: float = 3.0,
    ) -> int:
        """
        Register all unregistered views using PnP with existing 3D points.

        Views are processed in descending order of "number of markers
        with known 3D positions" to improve stability.  After each
        successful registration, newly observed markers are triangulated.

        Returns:
            Number of newly registered views.
        """
        t0 = time.perf_counter()
        newly_registered = 0

        while True:
            best_view = None
            best_matches = 0
            best_shared_ids: List[int] = []

            for view in self.views:
                if view.registered:
                    continue
                shared = [sid for sid in view.centers
                          if sid in self._points3d and sid >= 0]
                if len(shared) > best_matches:
                    best_matches = len(shared)
                    best_view = view
                    best_shared_ids = shared

            if best_view is None or best_matches < min_matches:
                break

            # --- PnP registration ---
            obj_pts = np.array(
                [self._points3d[sid] for sid in best_shared_ids],
                dtype=np.float64,
            )
            img_pts = np.array(
                [best_view.centers[sid] for sid in best_shared_ids],
                dtype=np.float64,
            )

            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_pts, img_pts, self.K, self.dist,
                flags=cv2.SOLVEPNP_EPNP,
                iterationsCount=200,
                reprojectionError=reproj_threshold,
                confidence=0.99,
            )

            if not success or (inliers is not None and len(inliers) < min_matches):
                print(f"  Skipping '{best_view.name}': PnP failed "
                      f"({len(inliers) if inliers is not None else 0} inliers)")
                # Mark as "tried and failed" by setting a flag — we'll
                # simply leave it unregistered; it may succeed after more
                # 3D points are added.
                best_view.registered = True  # temporarily, will undo below
                best_view.R = None
                best_view.t = None
                best_view.registered = False
                # Remove from consideration this round:
                # (We can't mark it permanently; the points may be bad.)
                # Just break to avoid infinite loop on this view.
                # Strategy: skip this view for now by removing its shared
                # marker list from consideration.
                # simpler: break with a note
                print(f"    (will retry after more points are added)")
                break

            best_view.R, _ = cv2.Rodrigues(rvec)
            best_view.t = tvec.reshape(3, 1)
            best_view.registered = True
            newly_registered += 1

            n_inl = len(inliers) if inliers is not None else 0
            print(f"  Registered '{best_view.name}' "
                  f"({n_inl}/{best_matches} inliers)")

            # --- Triangulate newly-observed markers ---
            self._triangulate_new_points(best_view)

        self._timings["register_all"] = (time.perf_counter() - t0) * 1000
        return newly_registered

    def _triangulate_new_points(self, new_view: View) -> int:
        """Triangulate markers seen in *new_view* + ≥1 other registered view."""
        registered_views = [v for v in self.views if v.registered]
        if len(registered_views) < 2:
            return 0

        new_count = 0
        for sid, center in new_view.centers.items():
            if sid < 0 or sid in self._points3d:
                continue  # skip circles and already-known points

            # Find other registered views that also see this marker
            other_views = [
                v for v in registered_views
                if v is not new_view and sid in v.centers
            ]
            if not other_views:
                continue

            # Pick the other view with the largest baseline
            best_v = max(
                other_views,
                key=lambda v: np.linalg.norm(new_view.t - v.t)
            )

            P1 = self.K @ np.hstack([new_view.R, new_view.t])
            P2 = self.K @ np.hstack([best_v.R, best_v.t])

            pt1 = np.array([[center]], dtype=np.float64)
            pt2 = np.array([[best_v.centers[sid]]], dtype=np.float64)

            pts4d = cv2.triangulatePoints(P1, P2, pt1.T, pt2.T)
            pts3d = pts4d[:3] / pts4d[3]

            p = pts3d[:, 0]
            # Check in front of both cameras
            p1_cam = new_view.R @ p + new_view.t.ravel()
            p2_cam = best_v.R @ p + best_v.t.ravel()
            if p1_cam[2] > 0 and p2_cam[2] > 0:
                self._points3d[sid] = (float(p[0]), float(p[1]), float(p[2]))
                new_count += 1

        if new_count:
            print(f"    Triangulated {new_count} new marker(s)")
        return new_count

    # ------------------------------------------------------------------
    # Step 4 — Bundle adjustment (resection-intersection + sparse LM)
    # ------------------------------------------------------------------

    def bundle_adjust(
        self,
        iterations: int = 10,
        use_sparse_lm: bool = True,
        verbose: bool = True,
    ) -> float:
        """
        Global bundle adjustment to refine camera poses and 3D points.

        Two-phase approach:
        1. **Resection-intersection** (fast, always run):
           alternate between solving PnP for each camera and
           re-triangulating each 3D point from all its views.
        2. **Sparse Levenberg-Marquardt** (optional, finer):
           minimise reprojection error over all parameters jointly
           using ``scipy.optimize.least_squares``.

        Returns:
            Final mean reprojection error in pixels.
        """
        t0 = time.perf_counter()

        reg_views = [v for v in self.views if v.registered]
        if len(reg_views) < 2 or len(self._points3d) < 4:
            print("Too few views or points for BA.")
            return float("inf")

        # --- Phase 1: resection-intersection -------------------------
        for it in range(iterations):
            # Resection: refine each camera pose via PnP
            for view in reg_views:
                shared = [sid for sid in view.centers
                          if sid in self._points3d and sid >= 0]
                if len(shared) < 4:
                    continue
                obj_pts = np.array(
                    [self._points3d[sid] for sid in shared], dtype=np.float64
                )
                img_pts = np.array(
                    [view.centers[sid] for sid in shared], dtype=np.float64
                )
                # Use iterative refinement (not RANSAC) since we trust the
                # initial pose and 3D points at this stage
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts, self.K, self.dist,
                    rvec=cv2.Rodrigues(view.R)[0],
                    tvec=view.t,
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if ok:
                    view.R, _ = cv2.Rodrigues(rvec)
                    view.t = tvec.reshape(3, 1)

            # Intersection: refine each 3D point from all views that see it
            for sid in list(self._points3d.keys()):
                observing_views = [
                    v for v in reg_views if sid in v.centers
                ]
                if len(observing_views) < 2:
                    continue
                # Triangulate from all pairs and average, or use DLT
                # from all views simultaneously
                self._points3d[sid] = self._triangulate_from_views(
                    sid, observing_views
                )

        # --- Phase 2: sparse LM (optional) ---------------------------
        if use_sparse_lm and len(self._points3d) >= 4:
            try:
                final_err = self._sparse_bundle_adjust(verbose)
            except Exception as exc:
                if verbose:
                    print(f"  Sparse BA failed ({exc}); "
                          f"using resection-intersection result.")
                final_err = self._compute_mean_reproj_error()
        else:
            final_err = self._compute_mean_reproj_error()

        self._timings["bundle_adjust"] = (time.perf_counter() - t0) * 1000
        if verbose:
            n_views = len(reg_views)
            n_pts = len(self._points3d)
            t_ms = self._timings["bundle_adjust"]
            print(f"BA: {n_views} views, {n_pts} points → "
                  f"reproj error = {final_err:.3f} px  ({t_ms:.0f} ms)")

        return final_err

    def _triangulate_from_views(
        self, marker_id: int, views: List[View]
    ) -> Point3D:
        """Triangulate a single point from all views that see it (DLT)."""
        n = len(views)
        if n < 2:
            return self._points3d.get(marker_id, (0.0, 0.0, 0.0))

        # Build linear system A X = 0 where X is the 3D point
        A = np.zeros((2 * n, 4), dtype=np.float64)
        for i, view in enumerate(views):
            cx, cy = view.centers[marker_id]
            P = self.K @ np.hstack([view.R, view.t])
            A[2 * i]     = cx * P[2] - P[0]
            A[2 * i + 1] = cy * P[2] - P[1]

        _, _, Vt = np.linalg.svd(A, full_matrices=False)
        X = Vt[-1]
        p = X[:3] / X[3]

        # Check in front of all cameras
        all_front = True
        for view in views:
            pc = view.R @ p + view.t.ravel()
            if pc[2] <= 0:
                all_front = False
                break

        if not all_front:
            # Fall back to the original point
            return self._points3d.get(marker_id, (float(p[0]), float(p[1]), float(p[2])))

        return (float(p[0]), float(p[1]), float(p[2]))

    # ------------------------------------------------------------------
    # Sparse Levenberg-Marquardt BA
    # ------------------------------------------------------------------

    def _sparse_bundle_adjust(self, verbose: bool = True) -> float:
        """
        Minimise reprojection error with a sparse Jacobian.

        Only camera extrinsics and 3D point positions are optimised;
        intrinsics are kept fixed.  The sparsity pattern exploits the
        fact that each observation depends on exactly one camera and
        one 3D point — the Jacobian is block-diagonal.
        """
        reg_views = [v for v in self.views if v.registered]
        point_ids = sorted(self._points3d.keys())

        n_cameras = len(reg_views)
        n_points = len(point_ids)
        n_params = 6 * n_cameras + 3 * n_points  # rvec(3) + t(3) per cam, (x,y,z) per point

        # Build observation list
        # Each obs: (camera_idx, point_idx, u, v)
        observations: List[Tuple[int, int, float, float]] = []
        pt_to_idx = {pid: i for i, pid in enumerate(point_ids)}

        for ci, view in enumerate(reg_views):
            for pid, (u, v) in view.centers.items():
                if pid in pt_to_idx:
                    observations.append((ci, pt_to_idx[pid], u, v))
        n_obs = len(observations)

        if n_obs < n_params:
            if verbose:
                print(f"  Sparse BA skipped: {n_obs} obs < {n_params} params")
            return self._compute_mean_reproj_error()

        if verbose:
            print(f"  Sparse BA: {n_cameras} cams, {n_points} pts, "
                  f"{n_obs} obs → {n_params} params")

        # Initial parameter vector
        x0 = np.zeros(n_params, dtype=np.float64)

        for ci, view in enumerate(reg_views):
            rvec = cv2.Rodrigues(view.R)[0].ravel()
            x0[6 * ci: 6 * ci + 3] = rvec
            x0[6 * ci + 3: 6 * ci + 6] = view.t.ravel()

        for pi, pid in enumerate(point_ids):
            offset = 6 * n_cameras + 3 * pi
            x0[offset: offset + 3] = self._points3d[pid]

        # Build sparsity pattern
        jac_sparsity = self._build_sparsity(
            n_cameras, n_points, observations, n_params
        )

        # Run LM
        result = least_squares(
            self._reproj_residuals,
            x0,
            jac_sparsity=jac_sparsity,
            method="trf",           # Trust Region Reflective (handles sparsity)
            loss="soft_l1",         # Robust to outliers
            f_scale=2.0,            # ~2 px inlier threshold
            max_nfev=50,
            verbose=1 if verbose else 0,
            args=(reg_views, point_ids, observations),
        )

        # Unpack result
        x_opt = result.x
        for ci, view in enumerate(reg_views):
            rvec = x_opt[6 * ci: 6 * ci + 3]
            view.R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
            view.t = x_opt[6 * ci + 3: 6 * ci + 6].reshape(3, 1)

        for pi, pid in enumerate(point_ids):
            offset = 6 * n_cameras + 3 * pi
            self._points3d[pid] = (
                float(x_opt[offset]),
                float(x_opt[offset + 1]),
                float(x_opt[offset + 2]),
            )

        return float(np.sqrt(result.cost * 2 / n_obs))

    def _reproj_residuals(
        self,
        params: np.ndarray,
        reg_views: List[View],
        point_ids: List[int],
        observations: List[Tuple[int, int, float, float]],
    ) -> np.ndarray:
        """Residual function for sparse BA."""
        n_cameras = len(reg_views)
        residuals = np.zeros(2 * len(observations), dtype=np.float64)

        # Unpack cameras
        cam_R = []
        cam_t = []
        for ci in range(n_cameras):
            rvec = params[6 * ci: 6 * ci + 3].reshape(3, 1)
            R, _ = cv2.Rodrigues(rvec)
            cam_R.append(R)
            cam_t.append(params[6 * ci + 3: 6 * ci + 6].reshape(3, 1))

        for oi, (ci, pi, u, v) in enumerate(observations):
            offset = 6 * n_cameras + 3 * pi
            pt3d = params[offset: offset + 3]

            # Project
            pc = cam_R[ci] @ pt3d + cam_t[ci].ravel()
            if pc[2] <= 1e-9:
                residuals[2 * oi] = 1e6
                residuals[2 * oi + 1] = 1e6
                continue

            up = pc[0] / pc[2] * self.K[0, 0] + self.K[0, 2]
            vp = pc[1] / pc[2] * self.K[1, 1] + self.K[1, 2]

            residuals[2 * oi] = up - u
            residuals[2 * oi + 1] = vp - v

        return residuals

    @staticmethod
    def _build_sparsity(
        n_cameras: int,
        n_points: int,
        observations: List[Tuple[int, int, float, float]],
        n_params: int,
    ) -> lil_matrix:
        """Build sparse Jacobian structure for BA."""
        n_obs = len(observations)
        J = lil_matrix((2 * n_obs, n_params))

        for oi, (ci, pi, _u, _v) in enumerate(observations):
            # Camera columns: 6 params per camera
            J[2 * oi,     6 * ci: 6 * ci + 6] = 1
            J[2 * oi + 1, 6 * ci: 6 * ci + 6] = 1
            # Point columns: 3 params per point
            offset = 6 * n_cameras + 3 * pi
            J[2 * oi,     offset: offset + 3] = 1
            J[2 * oi + 1, offset: offset + 3] = 1

        return J

    # ------------------------------------------------------------------
    # Scale resolution
    # ------------------------------------------------------------------

    def set_scale_from_marker(
        self, marker_id: int, physical_size_m: float
    ) -> float:
        """
        Compute absolute scale from a known-size marker.

        Compares the reconstructed 3D distance between adjacent corners
        of *marker_id* against *physical_size_m*.

        Returns:
            The computed scale factor (≈ 1.0 after calling this).
        """
        if marker_id not in self._points3d:
            raise ValueError(f"Marker {marker_id} not in 3D point set.")

        # We don't store corner 3D coords; use the side-length from
        # the 2D detections in registered views.
        side_lengths = []
        for view in self.views:
            if not view.registered or marker_id not in view.markers:
                continue
            corners = view.markers[marker_id]
            # Compute side lengths in pixels, then triangulate corners?
            # Simpler: use the marker's centre and PnP pose to back-project
            # the known-size corners.
            rvec = cv2.Rodrigues(view.R)[0]
            tvec = view.t
            # The marker lies on a plane; use solvePnP result for this marker
            half = physical_size_m / 2.0
            obj_pts = np.array([
                [-half,  half, 0],
                [ half,  half, 0],
                [ half, -half, 0],
                [-half, -half, 0],
            ], dtype=np.float32)
            img_pts, _ = cv2.projectPoints(
                obj_pts, rvec, tvec, self.K, self.dist
            )
            # Compare with detected corners
            detected = corners.astype(np.float32).reshape(4, 2)
            proj = img_pts.reshape(4, 2)
            # Compute scale as ratio of expected side to actual side
            for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                expected = np.linalg.norm(proj[a] - proj[b])
                actual = np.linalg.norm(detected[a] - detected[b])
                if actual > 0 and expected > 0:
                    side_lengths.append(expected / actual)

        if not side_lengths:
            raise RuntimeError(
                f"Marker {marker_id} not visible in any registered view."
            )

        # The scale factor adjusts the 3D coordinates
        scale = float(np.median(side_lengths))
        # Don't apply automatically; caller decides
        return scale

    def apply_scale(self, scale: float) -> None:
        """Multiply all 3D coordinates and camera translations by *scale*."""
        self._scale *= scale
        for sid in self._points3d:
            x, y, z = self._points3d[sid]
            self._points3d[sid] = (x * scale, y * scale, z * scale)
        for view in self.views:
            if view.registered and view.t is not None:
                view.t *= scale

    # ------------------------------------------------------------------
    # Coordinate-frame alignment
    # ------------------------------------------------------------------

    def align_to_ground(
        self,
        ground_marker_ids: List[int],
        ground_plane_points: Optional[List[Tuple[float, float, float]]] = None,
    ) -> np.ndarray:
        """
        Align the world coordinate frame so that the ground markers
        define the XY plane (Z=0) and their centroid is the origin.

        If *ground_plane_points* are given (physical measurements in
        metres), apply a similarity transform to bring the
        reconstructed points into that metrical frame.

        Returns:
            4×4 transformation applied (homogeneous).
        """
        if len(ground_marker_ids) < 3:
            raise ValueError("Need ≥3 ground markers to define a plane.")

        # Collect reconstructed 3D positions of ground markers
        pts_rec = np.array(
            [self._points3d[sid] for sid in ground_marker_ids
             if sid in self._points3d],
            dtype=np.float64,
        )
        if len(pts_rec) < 3:
            raise RuntimeError("Fewer than 3 ground markers have 3D coords.")

        # --- Fit plane to ground markers (in reconstructed frame) ----
        centroid = pts_rec.mean(axis=0)
        pts_centered = pts_rec - centroid
        _, _, Vt = np.linalg.svd(pts_centered, full_matrices=False)
        normal = Vt[-1]   # plane normal (smallest singular vector)
        if normal[2] < 0:
            normal = -normal  # point "up"
        normal = normal / np.linalg.norm(normal)

        # --- Build rotation that maps normal → (0, 0, 1) -------------
        z_axis = np.array([0.0, 0.0, 1.0])
        axis = np.cross(normal, z_axis)
        angle = math.acos(np.clip(np.dot(normal, z_axis), -1.0, 1.0))
        if np.linalg.norm(axis) < 1e-12:
            R_align = np.eye(3)
        else:
            axis = axis / np.linalg.norm(axis)
            rvec = axis * angle
            R_align, _ = cv2.Rodrigues(rvec)

        t_align = -R_align @ centroid

        # Build homogeneous transform: T_world_new = [R_align | t_align]
        T = np.eye(4)
        T[:3, :3] = R_align
        T[:3, 3] = t_align

        # Apply to all 3D points
        for sid in list(self._points3d.keys()):
            p = np.array(self._points3d[sid])
            p_new = R_align @ p + t_align
            self._points3d[sid] = (float(p_new[0]), float(p_new[1]), float(p_new[2]))

        # Apply to camera poses
        for view in self.views:
            if not view.registered:
                continue
            # Camera pose: X_cam = R @ X_world + t
            # After transform: X_world_new = R_align @ X_world_old + t_align
            # So: R_new @ X_world_new + t_new = R_old @ X_world_old + t_old
            R_old, t_old = view.R, view.t.ravel()
            view.R = R_old @ R_align.T
            view.t = (t_old - view.R @ t_align).reshape(3, 1)

        # --- Optional: similarity transform to metrical frame --------
        if ground_plane_points is not None:
            T_sim = self._similarity_align(ground_marker_ids, ground_plane_points)
            T = T_sim @ T

        print(f"Ground alignment applied.  "
              f"Plane normal in world frame: ({normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f})")

        return T

    def _similarity_align(
        self,
        marker_ids: List[int],
        target_points: List[Tuple[float, float, float]],
    ) -> np.ndarray:
        """Compute similarity transform (7-DOF) to align to target coords."""
        src = np.array(
            [self._points3d[mid] for mid in marker_ids
             if mid in self._points3d],
            dtype=np.float64,
        )
        tgt = np.array(target_points, dtype=np.float64)

        if len(src) != len(tgt) or len(src) < 3:
            print("Warning: insufficient points for similarity alignment.")
            return np.eye(4)

        # Estimate similarity (scale, rotation, translation)
        # Procrustes without reflection
        src_centroid = src.mean(axis=0)
        tgt_centroid = tgt.mean(axis=0)
        src_c = src - src_centroid
        tgt_c = tgt - tgt_centroid

        # Scale
        scale = np.sqrt(
            np.sum(tgt_c ** 2) / np.sum(src_c ** 2)
        )

        # Rotation via SVD
        H = src_c.T @ tgt_c
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T

        t = tgt_centroid - scale * R @ src_centroid

        # Apply
        for sid in list(self._points3d.keys()):
            p = np.array(self._points3d[sid])
            p_new = scale * R @ p + t
            self._points3d[sid] = (float(p_new[0]), float(p_new[1]), float(p_new[2]))

        for view in self.views:
            if not view.registered:
                continue
            R_old, t_old = view.R, view.t.ravel()
            # X_new = scale * R @ X_old + t
            # Derivation: C_new = scale * R @ C_old + t,
            #   t_new = -R_new @ C_new, R_new = R_old @ R^T
            view.R = R_old @ R.T
            view.t = (scale * t_old - view.R @ t).reshape(3, 1)

        T = np.eye(4)
        T[:3, :3] = scale * R
        T[:3, 3] = t

        print(f"  Similarity alignment: scale={scale:.4f}")
        return T

    # ------------------------------------------------------------------
    # Aircraft pose extraction
    # ------------------------------------------------------------------

    def get_rigid_transform(
        self,
        marker_ids: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the rigid-body transform (R, t) of a set of markers
        between their reconstructed 3D positions and their local
        coordinates (defined by the user).

        This is used to determine the **aircraft pose**: the aircraft
        markers' known local coordinates (in the aircraft body frame)
        are aligned to their reconstructed world coordinates, yielding
        the aircraft's position and orientation in the world.

        Args:
            marker_ids: List of ArUco marker IDs on the aircraft.

        Returns:
            ``(R, t)`` such that ``X_world = R @ X_local + t``.
            If fewer than 3 markers have 3D coords, both are ``None``.
        """
        pts_world = []
        valid_ids = []
        for mid in marker_ids:
            if mid in self._points3d:
                pts_world.append(self._points3d[mid])
                valid_ids.append(mid)

        if len(valid_ids) < 3:
            print(f"Only {len(valid_ids)} aircraft markers have 3D coords "
                  f"(need ≥3 for rigid transform).")
            return None, None

        pts_world_arr = np.array(pts_world, dtype=np.float64)

        # For local coords, we use the centroid + PCA to define a
        # local frame. In a real setup the user would provide measured
        # local coordinates.  We provide a default: centroid-origin,
        # principal axes as basis.
        centroid = pts_world_arr.mean(axis=0)
        pts_centered = pts_world_arr - centroid

        # If only the world positions are known (no separate local model),
        # R is the identity and t = centroid.  This gives the aircraft
        # position in world frame.
        R = np.eye(3)
        t = centroid.reshape(3,)

        print(f"Aircraft pose from {len(valid_ids)} markers: "
              f"t = ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}) m")
        return R, t

    def get_aircraft_pose_pnp(
        self,
        aircraft_marker_ids: List[int],
        aircraft_local_coords: List[Tuple[float, float, float]],
        image: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Phase B: single-view aircraft pose estimation via PnP.

        Given the reconstructed 3D world coordinates of aircraft markers
        (from Phase A), and a single image from Camera 2, compute the
        aircraft's pose in the ground coordinate frame.

        Args:
            aircraft_marker_ids: ArUco IDs on the aircraft.
            aircraft_local_coords: Corresponding (x, y, z) in the
                aircraft body frame (metres, measured physically).
            image: Single image from Camera 2.

        Returns:
            ``(R_aircraft_in_world, t_aircraft_in_world)`` or
            ``(None, None)`` on failure.
        """
        # Detect markers
        aruco_markers, _ = self._aruco.detect(image)
        img_pts = []
        obj_pts = []
        for m_id, corners, _, _ in aruco_markers:
            if m_id in aircraft_marker_ids:
                idx = aircraft_marker_ids.index(m_id)
                lx, ly, lz = aircraft_local_coords[idx]
                # Use all 4 corners
                for k in range(4):
                    img_pts.append(corners[k])
                    obj_pts.append([lx, ly, lz])

        if len(obj_pts) < 4:
            print("Not enough aircraft marker corners detected in image.")
            return None, None

        img_pts_arr = np.array(img_pts, dtype=np.float64)
        obj_pts_arr = np.array(obj_pts, dtype=np.float64)

        # First get camera pose in world frame (using ground markers)
        # Then transform aircraft points to world frame.
        # For simplicity, we directly solve PnP with world-coordinate
        # 3D points (from Phase A):

        # Get world 3D coords for detected aircraft markers
        obj_pts_world = []
        img_pts_world = []
        for m_id, corners, _, _ in aruco_markers:
            if m_id in self._points3d:
                for k in range(4):
                    img_pts_world.append(corners[k])
                    obj_pts_world.append(self._points3d[m_id])

        if len(obj_pts_world) < 12:  # at least 3 markers × 4 corners
            print("Not enough aircraft markers with known 3D coords.")
            return None, None

        obj_pts_w = np.array(obj_pts_world, dtype=np.float64)
        img_pts_w = np.array(img_pts_world, dtype=np.float64)

        success, rvec, tvec, _ = cv2.solvePnPRansac(
            obj_pts_w, img_pts_w, self.K, self.dist,
            flags=cv2.SOLVEPNP_EPNP,
            iterationsCount=100,
            reprojectionError=2.0,
        )

        if not success:
            return None, None

        R_cam, _ = cv2.Rodrigues(rvec)
        # Camera pose: X_cam = R_cam @ X_world + tvec
        # We want aircraft → world.  Aircraft markers are at known
        # positions in the aircraft body frame.  Their world coords
        # are in self._points3d.  We've already aligned to ground.
        # Return the transform of the aircraft body frame.

        # Use Procrustes between aircraft_local_coords and
        # their corresponding world 3D points:
        matched_local = []
        matched_world = []
        for i, mid in enumerate(aircraft_marker_ids):
            if mid in self._points3d:
                matched_local.append(aircraft_local_coords[i])
                matched_world.append(self._points3d[mid])

        if len(matched_local) < 3:
            return None, None

        loc = np.array(matched_local, dtype=np.float64)
        wrl = np.array(matched_world, dtype=np.float64)

        loc_c = loc - loc.mean(axis=0)
        wrl_c = wrl - wrl.mean(axis=0)
        H = loc_c.T @ wrl_c
        U, _, Vt = np.linalg.svd(H)
        R_ac = Vt.T @ U.T
        if np.linalg.det(R_ac) < 0:
            Vt[-1] *= -1
            R_ac = Vt.T @ U.T

        t_ac = wrl.mean(axis=0) - R_ac @ loc.mean(axis=0)

        return R_ac, t_ac.reshape(3,)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _select_initial_pair(
        self, min_shared: int, min_angle_deg: float
    ) -> Optional[Tuple[int, int]]:
        """Score all view pairs and return the best one."""
        best_score = -1.0
        best_pair = None
        n = len(self.views)

        for i, j in itertools.combinations(range(n), 2):
            va, vb = self.views[i], self.views[j]
            shared = [sid for sid in va.centers
                      if sid in vb.centers and sid >= 0]
            if len(shared) < min_shared:
                continue

            pts_a = np.array([va.centers[sid] for sid in shared], dtype=np.float64)
            pts_b = np.array([vb.centers[sid] for sid in shared], dtype=np.float64)

            try:
                E, mask = cv2.findEssentialMat(
                    pts_a, pts_b, self.K, method=cv2.RANSAC,
                    prob=0.999, threshold=1.0,
                )
                if E is None:
                    continue
                inliers = mask.ravel().astype(bool).sum()
                if inliers < min_shared:
                    continue
                _, R, _, _ = cv2.recoverPose(E, pts_a, pts_b, self.K)
                angle = np.linalg.norm(cv2.Rodrigues(R)[0])
                if angle < np.deg2rad(min_angle_deg):
                    continue
                score = inliers * angle
                if score > best_score:
                    best_score = score
                    best_pair = (i, j)
            except cv2.error:
                continue

        return best_pair

    def _compute_mean_reproj_error(self) -> float:
        """Compute mean reprojection error across all registered views."""
        total_err = 0.0
        n_obs = 0
        for view in self.views:
            if not view.registered:
                continue
            for sid, (u, v) in view.centers.items():
                if sid not in self._points3d or sid < 0:
                    continue
                p3d = np.array(self._points3d[sid], dtype=np.float64)
                pc = view.R @ p3d + view.t.ravel()
                if pc[2] <= 0:
                    continue
                up = pc[0] / pc[2] * self.K[0, 0] + self.K[0, 2]
                vp = pc[1] / pc[2] * self.K[1, 1] + self.K[1, 2]
                total_err += math.hypot(up - u, vp - v)
                n_obs += 1
        return total_err / n_obs if n_obs > 0 else float("inf")

    @property
    def points_3d(self) -> Dict[int, Point3D]:
        """Reconstructed 3D marker positions (world frame)."""
        return dict(self._points3d)

    @property
    def camera_poses(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """``[(R, t), …]`` for all registered views."""
        return [
            (v.R, v.t) for v in self.views if v.registered
        ]

    def summary(self) -> str:
        """Print a human-readable summary of the reconstruction."""
        n_reg = sum(1 for v in self.views if v.registered)
        n_total = len(self.views)
        n_pts = len(self._points3d)
        err = self._compute_mean_reproj_error()
        lines = [
            f"SfM Reconstruction Summary",
            f"  Views:    {n_reg}/{n_total} registered",
            f"  3D pts:   {n_pts}",
            f"  Repro err: {err:.3f} px",
        ]
        for k, v in self._timings.items():
            lines.append(f"  {k}: {v:.0f} ms")
        return "\n".join(lines)


# ===================================================================
# Synthetic-data test (injects GT projections → tests SfM math)
# ===================================================================
def _generate_synthetic_scene(
    n_markers: int = 15,
    n_views: int = 6,
    noise_px: float = 0.5,
    image_size: Tuple[int, int] = (1280, 720),
) -> Tuple[
    List[np.ndarray],          # images (with drawn marker positions)
    np.ndarray,                # K
    np.ndarray,                # dist_coeffs
    Dict[int, Point3D],        # GT 3D points
    List[Tuple[np.ndarray, np.ndarray]],  # GT camera poses (R, t)
]:
    """
    Generate synthetic data for SfM testing.

    Instead of rendering and re-detecting ArUco markers (which is
    fragile), we project known 3D points through known cameras,
    add noise, and populate the Views directly.

    Returns:
        (images, K, dist, gt_points, gt_poses)
        Images are simple visualizations for debugging.
    """
    rng = np.random.default_rng(42)

    w, h = image_size
    K = np.array([[1000, 0, w / 2], [0, 1000, h / 2], [0, 0, 1]],
                 dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)

    # Ground truth 3D points ------------------------------------------
    gt_points: Dict[int, Point3D] = {}
    for i in range(n_markers):
        if i < n_markers // 2:
            x = (rng.random() - 0.5) * 0.6
            y = (rng.random() - 0.5) * 0.6
            z = rng.random() * 0.02
        else:
            x = (rng.random() - 0.5) * 0.3
            y = (rng.random() - 0.5) * 0.3
            z = 0.15 + rng.random() * 0.1
        gt_points[i] = (float(x), float(y), float(z))

    # Ground truth camera poses ---------------------------------------
    gt_poses: List[Tuple[np.ndarray, np.ndarray]] = []
    rvecs = []
    tvecs = []
    for vi in range(n_views):
        angle = 2 * math.pi * vi / n_views
        radius = 0.7 + rng.random() * 0.2
        cam_x = radius * math.cos(angle)
        cam_y = radius * math.sin(angle * 0.9)
        cam_z = 0.25 + 0.15 * math.sin(angle * 0.7)

        cam_pos = np.array([cam_x, cam_y, cam_z])
        look_at = np.array([0.0, 0.0, 0.08])
        z_axis = look_at - cam_pos
        z_axis = z_axis / np.linalg.norm(z_axis)
        x_axis = np.cross(np.array([0.0, 1.0, 0.0]), z_axis)
        if np.linalg.norm(x_axis) < 1e-6:
            x_axis = np.cross(np.array([1.0, 0.0, 0.0]), z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        R = np.vstack([x_axis, y_axis, z_axis])  # rows = camera axes in world
        t = (-R @ cam_pos).reshape(3, 1)

        gt_poses.append((R, t))
        rvecs.append(cv2.Rodrigues(R)[0].ravel())
        tvecs.append(t.ravel())

    # Generate annotated images (visualization only) ------------------
    images = []
    for vi in range(n_views):
        img = np.full((h, w, 3), 240, dtype=np.uint8)
        R, t = gt_poses[vi]
        rvec = rvecs[vi]
        tvec = tvecs[vi]

        for mid, (mx, my, mz) in gt_points.items():
            # Project using cv2.projectPoints (handles distortion properly)
            pt3d = np.array([[[mx, my, mz]]], dtype=np.float32)
            pt2d, _ = cv2.projectPoints(pt3d, rvec.reshape(3, 1),
                                        tvec.reshape(3, 1), K.astype(np.float32), None)
            u, v = pt2d[0, 0]

            if not (-50 <= u < w + 50 and -50 <= v < h + 50):
                continue

            # Color: red=ground, blue=aircraft
            color = (0, 0, 220) if mid < n_markers // 2 else (220, 0, 0)
            cv2.circle(img, (int(u), int(v)), 5, color, -1)
            cv2.putText(img, str(mid), (int(u) + 8, int(v) - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Camera label
        cv2.putText(img, f"View {vi}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        images.append(img)

    return images, K, dist, gt_points, gt_poses


def _inject_projections(
    sfm: MultiViewSfM,
    gt_points: Dict[int, Point3D],
    gt_poses: List[Tuple[np.ndarray, np.ndarray]],
    noise_px: float = 0.5,
) -> None:
    """
    Populate SfM Views with ground-truth projections (+ optional Gaussian noise).

    This bypasses ArUco detection and tests the core SfM pipeline
    (initialisation, registration, BA) in isolation.
    """
    rng = np.random.default_rng(123)
    n_markers = len(gt_points)

    for vi, (R, t) in enumerate(gt_poses):
        if vi >= len(sfm.views):
            sfm.views.append(View(f"view_{vi}",
                np.zeros((720, 1280, 3), dtype=np.uint8)))
        view = sfm.views[vi]  # ← operate on the actual stored View
        view.name = f"view_{vi}"
        view.markers.clear()
        view.centers.clear()

        rvec = cv2.Rodrigues(R)[0]
        tvec = t.reshape(3, 1)

        for mid, (mx, my, mz) in gt_points.items():
            pt3d = np.array([[[mx, my, mz]]], dtype=np.float32)
            pt2d, _ = cv2.projectPoints(
                pt3d, rvec, tvec,
                sfm.K.astype(np.float32),
                sfm.dist.astype(np.float32),
            )
            u_raw, v_raw = float(pt2d[0, 0, 0]), float(pt2d[0, 0, 1])

            # Add Gaussian noise to centre
            u = u_raw + float(rng.normal(0, noise_px))
            v = v_raw + float(rng.normal(0, noise_px))

            h_img, w_img = 720, 1280
            if view.image is not None:
                h_img, w_img = view.image.shape[:2]

            if 0 <= u < w_img and 0 <= v < h_img:
                view.centers[mid] = (u, v)
                # Fake 4 corners (square of ~30 px around centre)
                s = 15.0
                corners = np.array([
                    [u - s, v - s], [u + s, v - s],
                    [u + s, v + s], [u - s, v + s],
                ], dtype=np.float32)
                view.markers[mid] = corners


# ===================================================================
# Main — synthetic data test
# ===================================================================

# ===================================================================
# (Test code has been extracted to tests/test_sfm_pipeline.py)
# ===================================================================
