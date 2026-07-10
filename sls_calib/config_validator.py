"""
Config validation — ensures camera configs match images and are self-consistent.

Usage:
  from sls_calib.config_validator import validate_camera_config, validate_image_size
"""
import yaml, numpy as np
from pathlib import Path
from typing import Tuple


def load_camera_config(config_path: str) -> Tuple[dict, np.ndarray, np.ndarray]:
    """Load and validate a camera config. Returns (config, K, dist)."""
    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    required = ['image_width', 'image_height', 'calibration']
    for key in required:
        if key not in cfg:
            raise ValueError(f"Camera config missing '{key}': {config_path}")

    cal = cfg['calibration']
    for k in ['fx', 'fy', 'cx', 'cy', 'dist']:
        if k not in cal:
            raise ValueError(f"Camera config missing 'calibration.{k}': {config_path}")

    # Validate field sanity
    w, h = cfg['image_width'], cfg['image_height']
    fx, fy = cal['fx'], cal['fy']
    cx, cy = cal['cx'], cal['cy']

    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image size: {w}x{h}")
    if fx <= 0 or fy <= 0:
        raise ValueError(f"Invalid focal length: fx={fx}, fy={fy}")
    if cx < 0 or cx > w or cy < 0 or cy > h:
        raise ValueError(f"Principal point outside image: cx={cx}, cy={cy}, image={w}x{h}")
    if len(cal['dist']) < 4:
        raise ValueError(f"Distortion coeffs too few: {len(cal['dist'])} (need >=4)")

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array(cal['dist'], dtype=np.float64)

    return cfg, K, dist


def validate_image_size(image, config_path: str):
    """Check that image dimensions match the camera config."""
    cfg, _, _ = load_camera_config(config_path)
    h, w = image.shape[:2]
    cw, ch = cfg['image_width'], cfg['image_height']
    if w != cw or h != ch:
        raise ValueError(
            f"Image size mismatch: image={w}x{h}, config={cw}x{ch}. "
            f"Config: {config_path}"
        )


def load_point_config(config_path: str, expected_unit: str = 'mm'
                      ) -> Tuple[dict, np.ndarray, list]:
    """Load a point config and validate unit. Returns (config, pts3d, pt_ids)."""
    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    unit = cfg.get('unit', '')
    if unit != expected_unit:
        raise ValueError(
            f"Point config unit mismatch: expected '{expected_unit}', "
            f"got '{unit}'. Config: {config_path}"
        )

    pts = cfg.get('points', {})
    pts3d = {}
    pt_ids = []
    for name, info in pts.items():
        if isinstance(info, list):
            pts3d[name] = np.array(info, dtype=np.float64)
        else:
            pts3d[name] = np.array([info['x_mm'], info['y_mm'], info['z_mm']],
                                   dtype=np.float64)
        pt_ids.append(name)

    if not pts3d:
        raise ValueError(f"No points found in: {config_path}")

    return cfg, pts3d, pt_ids
