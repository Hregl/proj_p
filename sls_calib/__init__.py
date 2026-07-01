"""
sls_calib —— SLS 相机标定、运动恢复结构（SfM）与基于标志点的姿态估计。

子模块
------
  marker_detector   基于亚像素精化的圆形标志点检测。
  camera_calib      基于 SLS 圆点网格的单相机内参标定。
  coded_marker      ArUco 编码标志点检测、PnP 姿态估计与标志点生成。
  sfm_pipeline      多视图 SfM 重建与光束法平差。
  stereo_calib      双相机立体标定与立体校正。
  pipeline          端到端标定流水线运行器。
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
