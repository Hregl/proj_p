"""
sls_calib — SLS camera calibration, SfM, and marker-based pose estimation.

Submodules
----------
  marker_detector   Circular marker detection with subpixel refinement.
  camera_calib      Single-camera intrinsic calibration via SLS dot grid.
  coded_marker      ArUco coded-marker detection, PnP, and marker generation.
  sfm_pipeline      Multi-view SfM reconstruction and bundle adjustment.
  stereo_calib      Dual-camera stereo calibration and rectification.
  pipeline          End-to-end calibration pipeline runner.
"""

# --- marker_detector ------------------------------------------------
from .marker_detector import SLSMarkerDetector
from .marker_detector import Marker

# --- camera_calib ---------------------------------------------------
from .camera_calib import CalibImage, Calibrator
from .camera_calib import Circle2D, World3D

# --- coded_marker ---------------------------------------------------
from .coded_marker import (
    ArUcoMarker,
    CodedMarkerDetector,
    UnifiedMarkerTracker,
    generate_charuco_board,
    generate_marker_image,
    generate_marker_sheet,
)

# --- sfm_pipeline ---------------------------------------------------
from .sfm_pipeline import MultiViewSfM, View
from .sfm_pipeline import Point3D

# --- stereo_calib ---------------------------------------------------
from .stereo_calib import StereoCalibrator, StereoParams, calibrate_stereo_rig

# --- pipeline -------------------------------------------------------
from .pipeline import CalibrationPipeline

__all__ = [
    # marker_detector
    "SLSMarkerDetector",
    "Marker",
    # camera_calib
    "CalibImage",
    "Calibrator",
    "Circle2D",
    "World3D",
    # coded_marker
    "ArUcoMarker",
    "CodedMarkerDetector",
    "UnifiedMarkerTracker",
    "generate_charuco_board",
    "generate_marker_image",
    "generate_marker_sheet",
    # sfm_pipeline
    "MultiViewSfM",
    "View",
    "Point3D",
    # stereo_calib
    "StereoCalibrator",
    "StereoParams",
    "calibrate_stereo_rig",
    # pipeline
    "CalibrationPipeline",
]
