"""
End-to-End Calibration Pipeline.

Orchestrates the full workflow:
  camera calibration → stereo calibration → SfM → pose estimation.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .marker_detector import SLSMarkerDetector
from .camera_calib import CalibImage, Calibrator
from .coded_marker import CodedMarkerDetector
from .sfm_pipeline import MultiViewSfM, View


class CalibrationPipeline:
    """
    One-stop entry point for the SLS calibration and pose-estimation
    workflow.

    Usage::

        pipeline = CalibrationPipeline({
            "data_dir": "data/my_scene",
            "output_dir": "output/my_scene",
            "marker_size_m": 0.025,
            "aruco_dict": "4x4_50",
            "ground_marker_ids": [0, 1, 2, 3],
            "aircraft_marker_ids": [4, 5, 6],
        })
        results = pipeline.run_all()
        pipeline.export_results()

    Parameters
    ----------
    config : dict
    """

    def __init__(self, config: dict) -> None:
        self.config = config

        # Paths
        self.data_dir = Path(config.get("data_dir", "data"))
        self.output_dir = Path(config.get("output_dir", "output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Marker config
        self.marker_size_m = config.get("marker_size_m")
        self.aruco_dict = config.get("aruco_dict", "4x4_50")
        self.ground_ids = config.get("ground_marker_ids", [])
        self.aircraft_ids = config.get("aircraft_marker_ids", [])

        # Camera intrinsics (populated by calibrate_camera / load)
        self._K: Optional[np.ndarray] = None
        self._dist: Optional[np.ndarray] = None
        self._image_size: Optional[Tuple[int, int]] = None

        # Pipeline state
        self._sfm: Optional[MultiViewSfM] = None
        self._aircraft_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._timings: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Step 1 — Camera intrinsic calibration
    # ------------------------------------------------------------------

    def calibrate_camera(
        self,
        calib_images: Optional[List[np.ndarray]] = None,
        circle_interval: float = 35.0,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Calibrate a single camera using SLS dot-grid images.

        Args:
            calib_images: List of calibration-board images.
            circle_interval: Physical circle spacing (mm).

        Returns:
            ``(K, dist_coeffs)``.
        """
        if calib_images is None:
            calib_images = self._load_images(
                self.data_dir / "calibration", "calib_*.png"
            )
        if not calib_images:
            print("No calibration images found.")
            return None, None

        t0 = time.perf_counter()
        calib_imgs = []
        for i, img in enumerate(calib_images):
            ci = CalibImage(name=f"calib_{i}", image=img, selected=True)
            calib_imgs.append(ci)

        calib = Calibrator()
        err = calib.extract_circles(calib_imgs, only_selected=True,
                                     smooth=True, debug=False)
        if err:
            print(f"Circle detection error: {err}")
            return None, None

        for ci in calib_imgs:
            err = ci.find_circle_indices(circle_interval, debug=False)
            if err:
                print(f"Grid assignment error for {ci.name}: {err}")

        report, K, dist = calib.calibrate_camera(calib_imgs, "calib", debug=True)
        print(report)

        if K is not None:
            self._K = K
            self._dist = dist
            if calib_images:
                h, w = calib_images[0].shape[:2]
                self._image_size = (w, h)

        self._timings["calibrate_camera"] = (time.perf_counter() - t0) * 1000
        return K, dist

    # ------------------------------------------------------------------
    # Step 2 — Load existing calibration
    # ------------------------------------------------------------------

    def load_calibration(self, npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load camera intrinsics from a ``.npz`` file.

        The file should contain ``camera_matrix`` and ``dist_coeffs``
        arrays (as saved by ``run_calibration.py`` or
        ``cv2.calibrateCamera``).
        """
        data = np.load(npz_path)
        self._K = data["camera_matrix"]
        self._dist = data["dist_coeffs"].ravel()
        return self._K, self._dist

    # ------------------------------------------------------------------
    # Step 3 — Multi-view SfM (Phase A)
    # ------------------------------------------------------------------

    def run_sfm(
        self,
        images: Optional[List[np.ndarray]] = None,
    ) -> Optional[MultiViewSfM]:
        """
        Run multi-view SfM reconstruction.

        Requires camera intrinsics to be set (via ``calibrate_camera``
        or ``load_calibration``).

        Args:
            images: List of scene images from Camera 1 (multiple views).

        Returns:
            ``MultiViewSfM`` with reconstructed 3D points and camera poses.
        """
        if self._K is None:
            print("Camera intrinsics not set. Run calibrate_camera() first.")
            return None

        if images is None:
            images = self._load_images(self.data_dir / "sfm", "view_*.png")
        if len(images) < 2:
            print("Need at least 2 images for SfM.")
            return None

        t0 = time.perf_counter()

        sfm = MultiViewSfM(
            self._K, self._dist,
            marker_size_m=self.marker_size_m,
            aruco_dict=self.aruco_dict,
        )
        sfm.add_views(images)

        if not sfm.initialize():
            print("SfM initialisation failed.")
            return None

        n_new = sfm.register_all()
        print(f"Registered {n_new} additional views "
              f"({sum(1 for v in sfm.views if v.registered)}/"
              f"{len(sfm.views)} total)")

        err = sfm.bundle_adjust(iterations=5, use_sparse_lm=True, verbose=True)
        print(f"BA reprojection error: {err:.3f} px")

        # Align to ground
        if self.ground_ids:
            sfm.align_to_ground(self.ground_ids)
            print(f"Aligned to {len(self.ground_ids)} ground markers")

        self._sfm = sfm
        self._timings["run_sfm"] = (time.perf_counter() - t0) * 1000
        print(sfm.summary())
        return sfm

    # ------------------------------------------------------------------
    # Step 4 — Aircraft pose estimation
    # ------------------------------------------------------------------

    def estimate_aircraft_pose(
        self,
        image: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Estimate aircraft pose (Phase B) via PnP against known 3D points.

        Requires SfM to have been run (``run_sfm``).

        Args:
            image: Single image from Camera 2 (arbitrary angle).

        Returns:
            ``(R_aircraft_in_world, t_aircraft_in_world)``.
        """
        if self._sfm is None:
            print("SfM not run yet. Call run_sfm() first.")
            return None
        if not self.aircraft_ids:
            print("aircraft_marker_ids not configured.")
            return None

        if image is None:
            images = self._load_images(self.data_dir / "inference", "cam2_*.png")
            if not images:
                print("No inference image found.")
                return None
            image = images[0]

        t0 = time.perf_counter()

        # For `get_aircraft_pose_pnp` we need local coords of aircraft markers.
        # Use the reconstructed 3D coords as "local" (proxy when no CAD model).
        local_coords = [
            self._sfm.points_3d.get(mid, (0.0, 0.0, 0.0))
            for mid in self.aircraft_ids
        ]

        R_ac, t_ac = self._sfm.get_aircraft_pose_pnp(
            self.aircraft_ids, local_coords, image,
        )

        if R_ac is None:
            print("Aircraft pose estimation failed.")
            return None

        self._aircraft_pose = (R_ac, t_ac)
        self._timings["estimate_aircraft_pose"] = (
            time.perf_counter() - t0
        ) * 1000

        # Euler angles for readability
        sy = np.sqrt(R_ac[0, 0] ** 2 + R_ac[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            rx = math.degrees(math.atan2(R_ac[2, 1], R_ac[2, 2]))
            ry = math.degrees(math.atan2(-R_ac[2, 0], sy))
            rz = math.degrees(math.atan2(R_ac[1, 0], R_ac[0, 0]))
        else:
            rx = math.degrees(math.atan2(-R_ac[1, 2], R_ac[1, 1]))
            ry = math.degrees(math.atan2(-R_ac[2, 0], sy))
            rz = 0.0

        print(f"Aircraft pose:")
        print(f"  t (m):    ({t_ac[0]:.4f}, {t_ac[1]:.4f}, {t_ac[2]:.4f})")
        print(f"  euler (°): roll={rx:.2f}  pitch={ry:.2f}  yaw={rz:.2f}")

        return R_ac, t_ac

    # ------------------------------------------------------------------
    # All-in-one
    # ------------------------------------------------------------------

    def run_all(
        self,
        calib_images: Optional[List[np.ndarray]] = None,
        sfm_images: Optional[List[np.ndarray]] = None,
        inference_image: Optional[np.ndarray] = None,
        calib_npz: Optional[str] = None,
    ) -> dict:
        """
        Run the full pipeline: calibrate → SfM → pose.

        Returns:
            Dictionary with keys: ``"K"``, ``"dist"``, ``"points_3d"``,
            ``"camera_poses"``, ``"aircraft_pose"``, ``"timings"``.
        """
        print("=" * 60)
        print("SLS Calibration Pipeline")
        print("=" * 60)

        # Camera calibration
        print("\n--- Step 1: Camera calibration ---")
        if calib_npz:
            self.load_calibration(calib_npz)
            print(f"Loaded intrinsics from {calib_npz}")
        else:
            K, dist = self.calibrate_camera(calib_images)
            if K is None:
                return {"error": "Camera calibration failed"}

        # SfM
        print("\n--- Step 2: Multi-view SfM ---")
        sfm = self.run_sfm(sfm_images)
        if sfm is None:
            return {"error": "SfM reconstruction failed"}

        # Aircraft pose
        print("\n--- Step 3: Aircraft pose estimation ---")
        pose = self.estimate_aircraft_pose(inference_image)
        if pose is None:
            return {"error": "Aircraft pose estimation failed"}

        print("\n" + "=" * 60)
        print("Pipeline complete.")
        for step, ms in self._timings.items():
            print(f"  {step}: {ms:.0f} ms")
        print("=" * 60)

        return {
            "K": self._K,
            "dist": self._dist,
            "points_3d": self._sfm.points_3d if self._sfm else {},
            "camera_poses": self._sfm.camera_poses if self._sfm else [],
            "aircraft_pose": pose,
            "timings": self._timings,
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_results(self) -> None:
        """Save all pipeline results to ``output_dir``."""
        if self._K is not None:
            np.savez(
                self.output_dir / "camera.npz",
                camera_matrix=self._K,
                dist_coeffs=self._dist,
            )

        if self._sfm is not None:
            pts = self._sfm.points_3d
            # Convert dict to arrays for saving
            ids = sorted(pts.keys())
            arr = np.array([pts[i] for i in ids], dtype=np.float64)
            np.savez(
                self.output_dir / "sfm_points.npz",
                marker_ids=np.array(ids),
                points_3d=arr,
            )

        if self._aircraft_pose is not None:
            R, t = self._aircraft_pose
            np.savez(
                self.output_dir / "aircraft_pose.npz",
                R=R, t=t.ravel(),
            )

        # Summary JSON
        summary = {
            "n_views": len(self._sfm.views) if self._sfm else 0,
            "n_registered": sum(1 for v in self._sfm.views if v.registered)
            if self._sfm else 0,
            "n_points_3d": len(self._sfm.points_3d) if self._sfm else 0,
            "timings_ms": self._timings,
        }
        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Results exported to {self.output_dir}/")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_images(directory: Path, glob_pattern: str) -> List[np.ndarray]:
        """Load images matching *glob_pattern* from *directory*."""
        if not directory.exists():
            return []
        paths = sorted(directory.glob(glob_pattern))
        images = []
        for p in paths:
            img = cv2.imread(str(p))
            if img is not None:
                images.append(img)
        return images
