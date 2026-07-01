# SLS Calibration Toolkit

Camera calibration, structure-from-motion (SfM), and aircraft pose estimation
using coded (ArUco) markers and circular dot targets.

## Project Structure

```
sls_calib/           # Python package (core algorithms)
  marker_detector.py   Circular marker detection + subpixel refinement
  camera_calib.py      Single-camera intrinsic calibration (SLS dot grid)
  coded_marker.py      ArUco coded markers: detection, PnP, generation
  sfm_pipeline.py      Multi-view SfM: reconstruction + bundle adjustment
  stereo_calib.py      Dual-camera stereo calibration + rectification
  pipeline.py          End-to-end pipeline runner

tools/               # CLI entry points
  generate_markers.py  Print ArUco marker sheets / ChArUco boards
  run_calibration.py   Single-camera intrinsic calibration
  run_sfm.py           Multi-view SfM reconstruction
  run_stereo_calib.py  Stereo rig calibration
  run_pipeline.py      Full pipeline (calibrate → SfM → pose)

tests/               # Accuracy / regression tests
data/                # Images and generated markers (not tracked)
output/              # Pipeline results (not tracked)
```

## Installation

```bash
git clone https://github.com/Hregl/proj_p.git
cd proj_p
python -m venv venv
venv\Scripts\activate         # Windows
# source venv/bin/activate    # Linux/macOS
pip install -r requirements.txt
```

Required: Python 3.10+, OpenCV 4.10+, NumPy 2.0+, SciPy 1.14+.

---

## Quick Verification

```bash
# Test marker detection on the sample calibration image
python tests/test_marker_detector.py data/p1.png
# Expected: 38 markers detected

# Test ArUco detection (generates a sheet, detects on it)
python tests/test_coded_marker.py
# Expected: 8/8 markers detected, PASS

# Test SfM accuracy on synthetic data
python tests/test_sfm_pipeline.py
# Expected: mean 3D error < 1 mm, PASS

# Test stereo calibration on synthetic data
python tests/test_stereo_calib.py
# Expected: rotation error < 0.5°, PASS
```

---

## End-to-End Workflow (Real Photos)

This is the complete pipeline for the aircraft pose estimation task:

```
Camera 1                       Camera 2
(multiple views)               (single arbitrary angle)
     │                              │
     ├─ SfM reconstruction ──┐      │
     │  (Phase A)            │      │
     │                       ▼      │
     │              3D marker coords│
     │              (ground truth)  │
     │                       │      │
     │                       └──────┤
     │                              │
     ▼                              ▼
   Aircraft pose ←──── PnP ──── Single image
   relative to ground         (Phase B)
```

### Step 0: Prepare Markers

1. **Geneate ArUco markers for printing:**

   ```bash
   python tools/generate_markers.py --dict 4x4_50 --ids 0 1 2 3 --size 300 -o markers_ground.png
   python tools/generate_markers.py --dict 4x4_50 --ids 4 5 6 7 8 --size 200 -o markers_aircraft.png
   ```

2. **Print** the markers on white paper (matte, not glossy — glossy paper reflects
   too much and causes detection failures).

3. **Attach** ground markers flat on the floor (define a coordinate reference).
   Attach aircraft markers to the model at known positions.

4. **(Optional) Generate a ChArUco calibration board:**

   ```bash
   python tools/generate_markers.py --charuco --squares 5 7 -o charuco_board.png
   ```

### Step 1: Calibrate Camera Intrinsics

You need the intrinsic parameters (focal length, principal point, distortion) of
each camera before running SfM or PnP.

**Option A — Using the SLS circular-dot target** (like `data/p1.png`):

Photograph the calibration target from 10-15 different angles (cover the entire
field of view, include tilted views).  Then:

```bash
python tools/run_calibration.py data/calibration/*.png --circle-interval 35 -o camera.npz
```

**Option B — Using a chessboard:**

Use the `run_stereo_calib.py` tool (which has built-in chessboard detection) to
calibrate each camera individually, or use OpenCV's `cv2.calibrateCamera` directly.

**Option C — Using a ChArUco board** (most robust if markers are partially occluded):

Same as chessboard — the `run_stereo_calib.py` tool supports `--pattern charuco`.

### Step 2: Multi-View SfM (Phase A — Ground Truth)

Use **Camera 1** to photograph the scene (ground + aircraft) from **6-15 different
angles** around the setup.  Ensure:

- Each photo sees **at least 4-6 ArUco markers** (both ground and aircraft)
- **~60-70% overlap** between consecutive views
- **Vary the height and angle** (not all from the same elevation)
- **Good lighting** — diffuse, even illumination; avoid direct reflections on markers
- **Sharp focus** — motion blur kills subpixel accuracy

Place the photos in `data/sfm/`, then:

```bash
python tools/run_sfm.py data/sfm/view_*.png \
    --camera camera.npz \
    --ground-ids 0 1 2 3 \
    --marker-size 0.03 \
    -o sfm_result.npz
```

**What happens:**
1. ArUco markers are detected in each image (IDs give automatic correspondence)
2. The best image pair is selected for initialisation
3. Essential matrix → relative pose → triangulation → initial 3D points
4. Remaining views are registered incrementally via PnP
5. Bundle adjustment refines all camera poses and 3D points simultaneously
6. Ground markers define the XY plane (Z=0); the world origin is their centroid

**Output:** `sfm_result.npz` containing:
- `marker_ids`: array of ArUco IDs
- `points_3d`: (N, 3) array of 3D marker positions in **metres** (world frame)
- `reproj_error`: final RMS reprojection error in pixels

An error < 1.0 px is excellent; < 2.0 px is acceptable.  Higher errors usually
mean poor calibration, motion blur, or insufficient angular coverage.

### Step 3: Aircraft Pose Estimation (Phase B — Inference)

Use **Camera 2** (which can be a different camera, also calibrated) to take a
**single photo** from any angle that captures both ground and aircraft markers.

```python
import cv2
import numpy as np
from sls_calib import CodedMarkerDetector, MultiViewSfM

# Load SfM results
sfm_data = np.load("sfm_result.npz", allow_pickle=True)
marker_ids = sfm_data["marker_ids"]
points_3d = {int(mid): tuple(p) for mid, p in zip(marker_ids, sfm_data["points_3d"])}

# Load Camera 2 calibration
cam2 = np.load("camera2.npz")
K2 = cam2["camera_matrix"]
dist2 = cam2["dist_coeffs"]

# Detect markers in the inference image
img = cv2.imread("data/inference/cam2_shot.png")
detector = CodedMarkerDetector("4x4_50")
markers, _ = detector.detect(img)

# PnP: camera pose from ground markers
# Use the known 3D positions from SfM
obj_pts = []
img_pts = []
for m_id, corners, center, _ in markers:
    if m_id in points_3d:
        obj_pts.append(points_3d[m_id])
        img_pts.append(center)

if len(obj_pts) >= 4:
    success, rvec, tvec, _ = cv2.solvePnPRansac(
        np.array(obj_pts, dtype=np.float32),
        np.array(img_pts, dtype=np.float32),
        K2, dist2,
        flags=cv2.SOLVEPNP_EPNP,
        iterationsCount=100,
        reprojectionError=2.0,
    )
    R_cam, _ = cv2.Rodrigues(rvec)
    print(f"Camera 2 position in world: {(-R_cam.T @ tvec).ravel()} m")
```

### Step 4 (Optional): Stereo Calibration

If you want both cameras to be geometrically related (for stereo depth or
cross-validation), calibrate them as a stereo pair:

```bash
python tools/run_stereo_calib.py \
    --left data/stereo/left_*.png \
    --right data/stereo/right_*.png \
    --pattern chessboard --pattern-size 9 6 --square-size 0.025 \
    -o stereo_params.npz
```

This computes the relative rotation R and translation T between the two cameras,
plus rectification maps for epipolar alignment.  The `--pattern` flag supports
`chessboard`, `circles` (SLS dot grid), and `charuco`.

### All-in-One (Config File)

Create a `config.json`:

```json
{
    "data_dir": "data/my_experiment",
    "output_dir": "output/my_experiment",
    "marker_size_m": 0.03,
    "aruco_dict": "4x4_50",
    "ground_marker_ids": [0, 1, 2, 3],
    "aircraft_marker_ids": [4, 5, 6, 7, 8]
}
```

Then run:

```bash
python tools/run_pipeline.py config.json
```

Or use pre-calibrated intrinsics:

```bash
python tools/run_pipeline.py --calib camera.npz \
    --sfm-images data/sfm/*.png \
    --ground-ids 0 1 2 3 --aircraft-ids 4 5 6 7 8 \
    --marker-size 0.03
```

---

## Marker Placement Guide

### Ground Markers (4-6 recommended)

- Place on a **flat, level surface** (floor or table)
- Spread them **30-80 cm apart** (wider = better angular resolution)
- **At least 1 marker should be offset** from the others (not all collinear)
  to prevent the ground plane normal from being ambiguous
- Use larger markers (5-10 cm) for better detection at distance
- ArUco IDs 0-3 are reserved for ground by convention

### Aircraft Markers (5-10 recommended)

- Distribute across the aircraft surface, **avoiding symmetry**
- At least 3 markers should be **non-collinear** (for rigid-body pose)
- Use smaller markers (2-5 cm) to fit on the model
- Markers on different surfaces (wing top, fuselage side, tail) provide the
  best pose constraints

### Photo Capture Tips

| Factor | Recommendation |
|---|---|
| Number of views (SfM) | 8-15 |
| Angular coverage | At least 120° around the scene |
| Overlap | 60-80% between consecutive views |
| Lighting | Diffuse, even; avoid direct sun or spot reflections |
| Focus | Sharp; motion blur is the #1 accuracy killer |
| Resolution | At least 20 px across the smallest marker |
| Camera settings | Fixed focus, fixed aperture, fixed white balance |

---

## Package API Reference

```python
from sls_calib import (
    # Marker detection
    SLSMarkerDetector,           # Circular dots (contour + subpixel)
    CodedMarkerDetector,         # ArUco coded markers with IDs
    UnifiedMarkerTracker,        # Combines both detectors

    # Calibration
    CalibImage, Calibrator,      # SLS dot-grid calibration

    # SfM
    MultiViewSfM, View,          # Multi-view reconstruction + BA

    # Stereo
    StereoCalibrator,            # Dual-camera calibration + rectification
    calibrate_stereo_rig,        # One-shot convenience function

    # Pipeline
    CalibrationPipeline,         # End-to-end orchestration

    # Marker generation
    generate_marker_image,       # Single ArUco marker
    generate_marker_sheet,       # Printable grid of markers
    generate_charuco_board,      # ChArUco calibration board
)
```

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| 0 ArUco markers detected | Poor lighting / blur / too small | Increase marker size, improve lighting, check focus |
| SfM init fails ("no suitable pair") | Not enough shared markers between views | Increase overlap, use more markers |
| High reprojection error (> 3 px) | Bad intrinsics or noisy detections | Recalibrate camera, check for motion blur |
| Ground plane normal wrong | Ground markers not coplanar or misidentified | Verify ground marker IDs, check flatness |
| Aircraft pose flips/jumps | Markers too symmetric | Add asymmetric marker placement |
| `ImportError` on `sls_calib` | Running from wrong directory | Always run from `d:/proj_p/` (project root) |

## References

- Hartley & Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed.
- OpenCV Documentation: [Camera Calibration](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- Garrido-Jurado et al., "Automatic generation and detection of highly reliable
  fiducial markers under occlusion", *Pattern Recognition*, 2014 (ArUco)
