"""
Ground coordinate system (G) pose estimation.

Estimates C_T_G (camera pose in ground frame) from 2D/3D ground-marker
correspondences using RANSAC PnP + LM refinement. Replaces the old
BoardPoseEstimator that relied on a calibration board to define the
world coordinate system.

Usage:
    from sls_calib.ground_pose import GroundPoseEstimator

    est = GroundPoseEstimator(K, dist, "configs/ground_markers_G.yaml")
    result = est.estimate(image, marker_2d_points)
    # result contains C_R_G, C_t_G, inlier_mask, rmse, per-point errors, etc.
"""

import json, yaml, time, math
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ====================================================================
# Structured result types
# ====================================================================

@dataclass
class GroundPoseResult:
    """Single-frame ground pose estimation result."""
    image_id: str = ""
    success: bool = False
    C_R_G: Optional[np.ndarray] = None   # Ground(G) → Camera(C) rotation
    C_t_G: Optional[np.ndarray] = None   # Ground(G) → Camera(C) translation (mm)
    rvec: Optional[np.ndarray] = None    # Rodrigues vector
    tvec: Optional[np.ndarray] = None    # translation vector (mm)

    # Quality metrics
    n_matched: int = 0          # total 2D/3D correspondences
    n_inliers: int = 0          # RANSAC inlier count
    inlier_ratio: float = 0.0   # n_inliers / n_matched
    rmse_px: float = 999.0      # inlier-only reprojection RMSE (pixels)
    max_error_px: float = 999.0 # max inlier reprojection error
    per_marker_errors: Dict[str, float] = field(default_factory=dict)

    # Failure info
    failure_reason: str = ""
    warnings: List[str] = field(default_factory=list)

    # Metadata
    timestamp: str = ""
    detection_count: int = 0    # number of 2D detections before matching

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        rv = self.rvec.ravel().tolist() if self.rvec is not None else []
        tv = self.tvec.ravel().tolist() if self.tvec is not None else []
        C_R_G_list = self.C_R_G.tolist() if self.C_R_G is not None else []
        return {
            'image_id': self.image_id,
            'success': self.success,
            'C_R_G': C_R_G_list,
            'C_t_G': self.C_t_G.tolist() if self.C_t_G is not None else [],
            'rvec': rv,
            'tvec': tv,
            'n_matched': self.n_matched,
            'n_inliers': self.n_inliers,
            'inlier_ratio': round(self.inlier_ratio, 4),
            'rmse_px': round(self.rmse_px, 4),
            'max_error_px': round(self.max_error_px, 4),
            'per_marker_errors': {k: round(v, 4) for k, v in self.per_marker_errors.items()},
            'failure_reason': self.failure_reason,
            'warnings': self.warnings,
            'timestamp': self.timestamp,
            'detection_count': self.detection_count,
        }


@dataclass
class GroundPoseReport:
    """Aggregated ground pose report for all frames."""
    session_id: str = ""
    stage: str = "ground_pose"
    status: str = "pending"  # pass / fail
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    n_total_frames: int = 0
    n_valid_frames: int = 0
    n_failed_frames: int = 0
    metrics: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)
    per_frame: List[dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    failed_items: List[str] = field(default_factory=list)


# ====================================================================
# GroundPoseEstimator
# ====================================================================

class GroundPoseEstimator:
    """Estimate camera-to-ground (C_T_G) pose from ground marker 2D/3D pairs.

    All tools (estimate_ground_pose, triangulate_aircraft_points,
    estimate_aircraft_pose) should use this class to ensure consistent
    PnP parameters, error reporting, and output format.
    """

    def __init__(self,
                 K: np.ndarray,
                 dist: np.ndarray,
                 ground_markers_yaml: str = "configs/ground_markers_G.yaml",
                 ransac_threshold_px: float = 2.0,
                 ransac_confidence: float = 0.99,
                 ransac_iterations: int = 200,
                 min_inliers: int = 4):
        """
        Args:
            K: Camera intrinsics (3x3).
            dist: Distortion coefficients.
            ground_markers_yaml: Path to YAML with ground marker 3D coords.
            ransac_threshold_px: RANSAC reprojection error threshold (px).
            ransac_confidence: RANSAC confidence level.
            ransac_iterations: RANSAC max iterations.
            min_inliers: Minimum inliers for a successful PnP solution.
        """
        self.K = np.asarray(K, dtype=np.float64)
        self.dist = np.asarray(dist, dtype=np.float64).ravel()

        # Load ground markers 3D
        with open(ground_markers_yaml, encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        cs = cfg.get('coordinate_system', '')
        if cs != 'G':
            raise ValueError(
                f"Ground markers must have coordinate_system='G', "
                f"got {cs!r}. File: {ground_markers_yaml}")

        unit = cfg.get('unit', 'mm')
        if unit != 'mm':
            raise ValueError(
                f"Ground markers unit must be mm, got {unit!r}. "
                f"File: {ground_markers_yaml}")

        self.ground_points_3d: Dict[str, np.ndarray] = {}
        for name, coords in cfg['points'].items():
            self.ground_points_3d[name] = np.array(coords, dtype=np.float64)

        self.point_ids = list(self.ground_points_3d.keys())
        self.ground_cfg = cfg

        # PnP parameters
        self.ransac_threshold_px = ransac_threshold_px
        self.ransac_confidence = ransac_confidence
        self.ransac_iterations = ransac_iterations
        self.min_inliers = min_inliers

    # ------------------------------------------------------------------
    def estimate(self,
                 image_id: str,
                 marker_2d: Dict[str, Tuple[float, float]],
                 detection_count: int = 0
                 ) -> GroundPoseResult:
        """Estimate C_T_G from 2D ground marker positions.

        Args:
            image_id: Identifier for this frame (e.g. filename stem).
            marker_2d: {marker_id: (u, v)} in pixel coordinates.
            detection_count: Total number of 2D points detected (before
                             matching to 3D), for diagnostics.

        Returns:
            GroundPoseResult with C_R_G, C_t_G, quality metrics.
        """
        result = GroundPoseResult(
            image_id=image_id,
            timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'),
            detection_count=detection_count,
        )

        # --- Build 3D/2D correspondences ---
        obj_pts, img_pts, matched_ids = [], [], []
        unmatched_ids = []
        for mid, uv in marker_2d.items():
            if mid in self.ground_points_3d:
                img_pts.append([uv[0], uv[1]])
                obj_pts.append(self.ground_points_3d[mid].tolist())
                matched_ids.append(mid)
            else:
                unmatched_ids.append(mid)

        result.n_matched = len(obj_pts)

        if result.n_matched < self.min_inliers:
            result.failure_reason = (
                f"Insufficient matches: {result.n_matched} "
                f"(need >= {self.min_inliers}). "
                f"Unmatched IDs: {unmatched_ids}")
            return result

        if unmatched_ids:
            result.warnings.append(
                f"{len(unmatched_ids)} 2D points have no 3D match: "
                f"{unmatched_ids[:5]}{'...' if len(unmatched_ids) > 5 else ''}")

        obj_arr = np.array(obj_pts, dtype=np.float64)
        img_arr = np.array(img_pts, dtype=np.float64)

        # --- RANSAC PnP ---
        # EPNP handles non-planar points; IPPE is for planar objects only
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_arr, img_arr, self.K, self.dist,
            flags=cv2.SOLVEPNP_EPNP,
            iterationsCount=self.ransac_iterations,
            reprojectionError=self.ransac_threshold_px,
            confidence=self.ransac_confidence)

        if not success or inliers is None or len(inliers) < self.min_inliers:
            # Fallback: iterative PnP without RANSAC
            try:
                success2, rvec2, tvec2 = cv2.solvePnP(
                    obj_arr, img_arr, self.K, self.dist,
                    flags=cv2.SOLVEPNP_ITERATIVE)
                if success2:
                    rvec, tvec = rvec2, tvec2
                    inliers = None
                else:
                    result.failure_reason = (
                        f"PnP failed: RANSAC returned {len(inliers) if inliers is not None else 0} "
                        f"inliers, and iterative fallback also failed")
                    return result
            except cv2.error as e:
                result.failure_reason = f"PnP exception: {e}"
                return result

        # --- LM refinement on inliers (or all points if no inlier mask) ---
        if inliers is not None and len(inliers) >= self.min_inliers:
            inl_idx = inliers.ravel()
            inl_mask = np.zeros(len(obj_arr), dtype=bool)
            inl_mask[inl_idx] = True
            try:
                rvec2, tvec2 = cv2.solvePnPRefineLM(
                    obj_arr[inl_mask], img_arr[inl_mask],
                    self.K, self.dist, rvec, tvec,
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
                rvec, tvec = rvec2, tvec2
            except cv2.error:
                pass  # Keep RANSAC result if LM fails
        else:
            inl_mask = np.ones(len(obj_arr), dtype=bool)

        # --- Compute reprojection errors ---
        if inliers is not None and len(inliers) >= self.min_inliers:
            # RMSE on inliers only
            proj_inl, _ = cv2.projectPoints(
                obj_arr[inl_mask], rvec, tvec, self.K, self.dist)
            errs_inl = np.linalg.norm(
                proj_inl.reshape(-1, 2) - img_arr[inl_mask], axis=1)
            result.rmse_px = float(np.sqrt(np.mean(errs_inl ** 2)))
            result.max_error_px = float(np.max(errs_inl))
            result.n_inliers = int(np.sum(inl_mask))
        else:
            proj_all, _ = cv2.projectPoints(
                obj_arr, rvec, tvec, self.K, self.dist)
            errs_all = np.linalg.norm(
                proj_all.reshape(-1, 2) - img_arr, axis=1)
            result.rmse_px = float(np.sqrt(np.mean(errs_all ** 2)))
            result.max_error_px = float(np.max(errs_all))
            result.n_inliers = result.n_matched
            inl_mask = np.ones(len(obj_arr), dtype=bool)

        result.inlier_ratio = result.n_inliers / result.n_matched if result.n_matched > 0 else 0.0

        # Per-marker errors
        proj_all, _ = cv2.projectPoints(obj_arr, rvec, tvec, self.K, self.dist)
        for i, mid in enumerate(matched_ids):
            err = float(np.linalg.norm(proj_all[i, 0] - img_arr[i]))
            result.per_marker_errors[mid] = err

        # --- Extract C_T_G ---
        C_R_G, _ = cv2.Rodrigues(rvec)
        result.C_R_G = C_R_G
        result.C_t_G = tvec.ravel().astype(np.float64)
        result.rvec = rvec
        result.tvec = tvec
        result.success = True

        # Quality warnings
        if result.rmse_px > 3.0:
            result.warnings.append(
                f"High reprojection RMSE: {result.rmse_px:.2f} px (> 3.0)")
        if result.inlier_ratio < 0.7:
            result.warnings.append(
                f"Low inlier ratio: {result.inlier_ratio:.1%} (< 70%)")

        return result

    # ------------------------------------------------------------------
    def process_image_set(self,
                          image_paths: List[str],
                          annotations_dir: str = "annotations/ground_2d",
                          output_poses_json: Optional[str] = None,
                          output_report_json: Optional[str] = None
                          ) -> GroundPoseReport:
        """Process a batch of images and output structured results.

        Args:
            image_paths: List of image file paths.
            annotations_dir: Directory with per-image YAML annotation files
                             (each file has {marker_id: {pixel_x, pixel_y, visible}}).
            output_poses_json: If set, write per-frame poses to this JSON file.
            output_report_json: If set, write aggregated report to this JSON file.

        Returns:
            GroundPoseReport with aggregated quality metrics.
        """
        report = GroundPoseReport(
            session_id=Path(output_poses_json or "unknown").stem,
            inputs={
                'n_images': len(image_paths),
                'n_ground_markers': len(self.ground_points_3d),
            },
            thresholds={
                'ransac_reprojection_px': self.ransac_threshold_px,
                'min_inliers': self.min_inliers,
            },
            n_total_frames=len(image_paths),
        )

        per_frame_results = []
        for img_path in image_paths:
            stem = Path(img_path).stem
            ann_path = Path(annotations_dir) / f"{stem}_points.yaml"

            detection_count = 0
            marker_2d = {}
            if ann_path.exists():
                with open(ann_path, encoding='utf-8') as f:
                    ann_data = yaml.safe_load(f)
                for name, info in ann_data.get('points', {}).items():
                    px = float(info.get('pixel_x', -1))
                    py = float(info.get('pixel_y', -1))
                    visible = info.get('visible', True)
                    detection_count += 1
                    if px >= 0 and py >= 0 and visible:
                        marker_2d[name] = (px, py)
            else:
                # No annotation file — try to auto-detect markers
                img = cv2.imread(img_path)
                if img is None:
                    report.failed_items.append(f"{stem}: cannot read image")
                    continue

            result = self.estimate(stem, marker_2d, detection_count)
            per_frame_results.append(result)

            if result.success:
                report.n_valid_frames += 1
            else:
                report.n_failed_frames += 1
                report.failed_items.append(
                    f"{stem}: {result.failure_reason}")

            for w in result.warnings:
                if w not in report.warnings:
                    report.warnings.append(w)

        # --- Aggregate metrics ---
        valid_rmse = [r.rmse_px for r in per_frame_results if r.success]
        valid_inliers = [r.n_inliers for r in per_frame_results if r.success]
        valid_ratios = [r.inlier_ratio for r in per_frame_results if r.success]

        if valid_rmse:
            report.metrics = {
                'rmse_px_mean': round(float(np.mean(valid_rmse)), 3),
                'rmse_px_median': round(float(np.median(valid_rmse)), 3),
                'rmse_px_max': round(float(np.max(valid_rmse)), 3),
                'rmse_px_min': round(float(np.min(valid_rmse)), 3),
                'inlier_count_mean': round(float(np.mean(valid_inliers)), 1),
                'inlier_ratio_mean': round(float(np.mean(valid_ratios)), 3),
            }
            report.status = 'pass' if report.n_failed_frames == 0 else 'partial'
        else:
            report.metrics = {}
            report.status = 'fail'

        report.per_frame = [r.to_dict() for r in per_frame_results]
        report.outputs = {
            'valid_frames': report.n_valid_frames,
            'failed_frames': report.n_failed_frames,
        }

        # --- Write outputs ---
        if output_poses_json:
            Path(output_poses_json).parent.mkdir(parents=True, exist_ok=True)
            # Write a flat list of per-frame poses for programmatic access
            poses_data = {
                'coordinate_system': 'G',
                'unit': 'mm',
                'poses': {r.image_id: {
                    'C_R_G': r.C_R_G.tolist() if r.C_R_G is not None else None,
                    'C_t_G': r.C_t_G.tolist() if r.C_t_G is not None else None,
                    'rmse_px': r.rmse_px,
                    'n_inliers': r.n_inliers,
                    'success': r.success,
                } for r in per_frame_results},
            }
            with open(output_poses_json, 'w', encoding='utf-8') as f:
                json.dump(poses_data, f, indent=2, ensure_ascii=False)

        if output_report_json:
            Path(output_report_json).parent.mkdir(parents=True, exist_ok=True)
            report_data = {
                'session_id': report.session_id,
                'stage': report.stage,
                'status': report.status,
                'inputs': report.inputs,
                'outputs': report.outputs,
                'metrics': report.metrics,
                'thresholds': report.thresholds,
                'warnings': report.warnings,
                'failed_items': report.failed_items,
                'per_frame': report.per_frame,
            }
            with open(output_report_json, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)

        return report

    # ------------------------------------------------------------------
    def get_G_R_C(self, result: GroundPoseResult
                  ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get G_R_C, G_t_C (camera→world) from a C_T_G result.

        G_T_C = inv(C_T_G): X_G = G_R_C * X_C + G_t_C
        """
        if not result.success or result.C_R_G is None:
            return None, None
        G_R_C = np.linalg.inv(result.C_R_G)
        G_t_C = -G_R_C @ result.C_t_G
        return G_R_C, G_t_C


# ====================================================================
# Convenience function for loading ground poses from stage 2 output
# ====================================================================

def load_ground_poses(json_path: str) -> Dict[str, dict]:
    """Load stage 2 ground pose output into a dict keyed by image_id.

    Returns:
        {image_id: {'C_R_G': ndarray, 'C_t_G': ndarray, 'rmse_px': float, ...}, ...}
    """
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    poses = {}
    for img_id, pdata in data.get('poses', {}).items():
        if pdata.get('success', False) and pdata.get('C_R_G') is not None:
            poses[img_id] = {
                'C_R_G': np.array(pdata['C_R_G'], dtype=np.float64),
                'C_t_G': np.array(pdata['C_t_G'], dtype=np.float64),
                'rmse_px': pdata.get('rmse_px', 999),
                'n_inliers': pdata.get('n_inliers', 0),
            }
    return poses
