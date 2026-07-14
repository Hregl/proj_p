"""
SfM-based aircraft marker 3D reconstruction.

Uses fundamental/essential matrix from 2D marker correspondences
across views. Camera poses are recovered from E-matrix decomposition
(not board PnP). The calibration board provides only absolute scale.

Flow:
  1. Load 2D annotations for N views
  2. Pairwise F-matrix → E-matrix → (R,t) decomposition
  3. Triangulate markers across all view pairs
  4. Use board PnP distance to determine scale factor

Advantage over triangulate_aircraft_points.py:
  - Does NOT require board PnP for camera pose
  - Works even when the board is poorly detected
  - Only needs 2+ views with marker annotations

Usage:
  python tools/sfm_aircraft.py data/scene_20mm1/*.bmp \
      --point-names Cockpit L_Wingtip R_Wingtip Spine \
                     L_HTail R_HTail L_VTail R_VTail \
      --config configs/cameras/camera_25mm_far.yaml
"""
import sys, cv2, yaml, numpy as np, math, os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.triangulate_aircraft_points import BoardPoseEstimator


class SfMReconstructor:
    """SfM-based 3D reconstruction using E-matrix decomposition."""

    def __init__(self, K: np.ndarray, dist: np.ndarray):
        self.K = K.astype(np.float64)
        self.dist = np.asarray(dist, dtype=np.float64).ravel()
        self.point_names: List[str] = []
        # observations[img_idx] = {name: (u, v), ...}
        self.observations: Dict[int, Dict[str, Tuple[float, float]]] = {}
        # Camera poses relative to view 0: {img_idx: (R, t)}
        self.poses: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.points3d: Dict[str, np.ndarray] = {}
        self.scale = 1.0

    def load_annotations(self, annotation_files: List[str],
                         point_names: List[str]):
        """Load 2D marker annotations from YAML files."""
        self.point_names = list(point_names)

        for i, yaml_path in enumerate(annotation_files):
            with open(yaml_path, encoding='utf-8') as f:
                data = yaml.safe_load(f)

            obs = {}
            for name in point_names:
                if name in data.get('points', {}):
                    p = data['points'][name]
                    px, py = float(p['pixel_x']), float(p['pixel_y'])
                    if px >= 0 and py >= 0:
                        obs[name] = (px, py)

            if len(obs) >= 6:  # need at least 6 points for F matrix
                self.observations[i] = obs
                print(f'  View {i}: {Path(yaml_path).stem} -> {len(obs)} points')

        print(f'  Loaded {len(self.observations)} valid views')

    def compute_epipolar_geometry(self, view_a: int, view_b: int
                                   ) -> Optional[Tuple[np.ndarray, np.ndarray,
                                                       np.ndarray]]:
        """Compute F, E, and (R,t) between two views. Returns (F, R, t) or None."""
        common = [n for n in self.point_names
                  if n in self.observations[view_a]
                  and n in self.observations[view_b]]

        if len(common) < 8:
            return None

        pts_a = np.array([self.observations[view_a][n] for n in common],
                         dtype=np.float64)
        pts_b = np.array([self.observations[view_b][n] for n in common],
                         dtype=np.float64)

        # Normalize for numerical stability
        pts_a_norm = cv2.undistortPoints(pts_a.reshape(-1, 1, 2),
                                          self.K, self.dist)
        pts_b_norm = cv2.undistortPoints(pts_b.reshape(-1, 1, 2),
                                          self.K, self.dist)

        # Essential matrix from normalized points
        E, mask = cv2.findEssentialMat(
            pts_a_norm.reshape(-1, 2), pts_b_norm.reshape(-1, 2),
            focal=1.0, pp=(0.0, 0.0),
            method=cv2.RANSAC, prob=0.999, threshold=0.001)

        if E is None or mask is None or mask.sum() < 6:
            return None

        # Recover pose (R, t) from E
        inlier_pts_a = pts_a_norm.reshape(-1, 2)[mask.ravel() == 1]
        inlier_pts_b = pts_b_norm.reshape(-1, 2)[mask.ravel() == 1]

        _, R, t, _ = cv2.recoverPose(E, inlier_pts_a, inlier_pts_b)

        # Also compute F for reference
        F, _ = cv2.findFundamentalMat(pts_a, pts_b, cv2.FM_RANSAC,
                                       3.0, 0.99)

        return F, R, t

    def triangulate_from_pair(self, view_a: int, view_b: int,
                               R_ab: np.ndarray, t_ab: np.ndarray
                               ) -> Dict[str, np.ndarray]:
        """Triangulate marker points from a view pair.
        view_a is treated as origin (I, 0), view_b at (R_ab, t_ab).
        """
        P_a = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P_b = self.K @ np.hstack([R_ab, t_ab.reshape(3, 1)])

        points = {}
        for name in self.point_names:
            if (name in self.observations[view_a] and
                    name in self.observations[view_b]):
                ua, va = self.observations[view_a][name]
                ub, vb = self.observations[view_b][name]

                pts4d = cv2.triangulatePoints(
                    P_a, P_b,
                    np.array([[ua], [va]], dtype=np.float32),
                    np.array([[ub], [vb]], dtype=np.float32))
                p3d = pts4d[:3, 0] / pts4d[3, 0]

                # Check in front of both cameras
                if p3d[2] > 0 and (R_ab @ p3d + t_ab.ravel())[2] > 0:
                    points[name] = p3d

        return points

    def calibrate_scale(self, board_estimator: BoardPoseEstimator,
                        image_paths: List[str],
                        view_a: int, view_b: int
                        ) -> float:
        """Determine scale by comparing E-matrix baseline to board-PnP baseline.

        The E-matrix gives (R, t) with |t|=1 (unit baseline).
        Board PnP gives camera positions in mm → actual baseline in mm.
        Scale = actual_baseline_mm (since E-matrix baseline = 1).
        """
        # Get board PnP camera positions for the two views
        pos_mm = []
        for vi in [view_a, view_b]:
            img = cv2.imread(image_paths[vi])
            if img is None:
                continue
            G_R_C, G_t_C, rmse, _, _ = board_estimator.process_image(img)
            if G_R_C is None:
                print(f'  View {vi}: board PnP FAILED')
                continue
            # Camera position in board frame (mm)
            pos_mm.append(G_t_C)
            print(f'  View {vi}: cam pos = ({G_t_C[0]:.0f},{G_t_C[1]:.0f},{G_t_C[2]:.0f}) mm, RMSE={rmse:.3f}px')

        if len(pos_mm) < 2:
            # Fallback: use median board distance
            dists = []
            for vi in self.observations:
                img = cv2.imread(image_paths[vi])
                if img is None: continue
                G_R_C, G_t_C, rmse, _, _ = board_estimator.process_image(img)
                if G_R_C is not None:
                    dists.append(float(np.linalg.norm(G_t_C)))
            self.scale = float(np.median(dists)) if dists else 500.0
            print(f'  Fallback scale: {self.scale:.0f} mm')
            return self.scale

        # Baseline in mm
        baseline_mm = float(np.linalg.norm(pos_mm[1] - pos_mm[0]))
        # E-matrix |t| = 1, so scale = baseline_mm
        self.scale = baseline_mm
        print(f'  Baseline: {baseline_mm:.0f} mm -> scale = {self.scale:.0f}')
        return self.scale

    def reconstruct(self, image_paths: List[str],
                    board_estimator: BoardPoseEstimator
                    ) -> Dict[str, np.ndarray]:
        """Main reconstruction pipeline."""
        view_indices = sorted(self.observations.keys())

        if len(view_indices) < 2:
            print("Need at least 2 valid views")
            return {}

        # --- Step 1: Compute pair-wise (R,t) from E matrix ---
        print(f"\nStep 1: Epipolar geometry (view pairs)")
        pair_results = []
        for i in range(len(view_indices)):
            for j in range(i + 1, len(view_indices)):
                a, b = view_indices[i], view_indices[j]
                result = self.compute_epipolar_geometry(a, b)
                if result is not None:
                    F, R, t = result
                    n_common = len([n for n in self.point_names
                                    if n in self.observations[a]
                                    and n in self.observations[b]])
                    print(f'  {a}-{b}: {n_common} common, E-matrix OK')
                    pair_results.append((a, b, R, t, n_common))
                else:
                    n_common = len([n for n in self.point_names
                                    if n in self.observations[a]
                                    and n in self.observations[b]])
                    if n_common >= 6:
                        print(f'  {a}-{b}: {n_common} common, E-matrix FAILED')

        if not pair_results:
            print("No valid view pairs found")
            return {}

        # --- Step 2: Triangulate from best pair WITH board PnP for scale ---
        # Pre-compute board PnP for all views
        board_poses = {}  # {view_idx: G_t_C}
        for vi in view_indices:
            img = cv2.imread(image_paths[vi])
            if img is not None:
                G_R_C, G_t_C, rmse, _, _ = board_estimator.process_image(img)
                if G_R_C is not None:
                    board_poses[vi] = G_t_C

        # Pick best pair where BOTH have board PnP
        pair_results.sort(key=lambda x: -x[4])
        best_pair = None
        for a, b, R, t, n in pair_results:
            if a in board_poses and b in board_poses:
                best_pair = (a, b, R, t, n)
                break

        if best_pair is None:
            # Fallback: use any pair, scale from median board distance
            best_pair = pair_results[0]
            best_a, best_b, R_best, t_best, _ = best_pair
            dists = [float(np.linalg.norm(p)) for p in board_poses.values()]
            self.scale = float(np.median(dists)) if dists else 500.0
            print(f"\nStep 2: No pair with dual board PnP, fallback scale={self.scale:.0f}mm")
        else:
            best_a, best_b, R_best, t_best, _ = best_pair
            baseline_mm = float(np.linalg.norm(board_poses[best_b] - board_poses[best_a]))
            self.scale = baseline_mm
            print(f"\nStep 2: Best pair {best_a}-{best_b}, baseline={baseline_mm:.0f}mm")

        print(f"Triangulating...")
        raw_points = self.triangulate_from_pair(best_a, best_b, R_best, t_best)
        print(f'  {len(raw_points)} points triangulated')

        # --- Step 3: Multi-view refinement ---
        print(f"\nStep 3: Multi-view refinement")
        # Merge triangulations from additional pairs
        merged = dict(raw_points)
        for a, b, R, t, _ in pair_results[1:]:
            pts = self.triangulate_from_pair(a, b, R, t)
            for name, p in pts.items():
                if name not in merged:
                    merged[name] = p
                else:
                    merged[name] = (merged[name] + p) / 2.0  # average

        # Apply scale
        self.points3d = {name: p * self.scale for name, p in merged.items()}

        return self.points3d


# ====================================================================
# GUI for labeling frames (reuse from triangulation tool)
# ====================================================================

def annotate_frames_gui(images: List[np.ndarray],
                        image_paths: List[str],
                        point_names: List[str],
                        existing_labels: Optional[Dict[int, str]] = None
                        ) -> Dict[int, Dict[str, Tuple[float, float]]]:
    """Interactive GUI to label marker points across multiple frames.
    Returns {img_idx: {name: (u, v), ...}}
    """
    from tools.triangulate_aircraft_points import AircraftTriangulationGUI

    # Create a minimal board estimator (just for the GUI, not for actual PnP)
    K_dummy = np.eye(3)
    dist_dummy = np.zeros(5)
    board_est = BoardPoseEstimator(K_dummy, dist_dummy)

    gui = AircraftTriangulationGUI(images, image_paths, board_est,
                                    K_dummy, dist_dummy)
    gui.point_names = list(point_names)
    gui.point_count = len(point_names)
    for i in range(gui.point_count):
        gui.observations[i] = []

    # Mark all frames as "valid"
    gui.valids = [True] * len(images)
    gui.poses = [{'rmse': 0} for _ in images]

    # Label frame 0
    if not gui.label_first_frame():
        return {}

    # Label remaining frames
    gui.annotate_remaining_frames()

    # Convert observations to our format
    result: Dict[int, Dict[str, Tuple[float, float]]] = {}
    for img_idx in range(len(images)):
        obs = {}
        for pt_idx in range(gui.point_count):
            for obs_tuple in gui.observations.get(pt_idx, []):
                if obs_tuple[0] == img_idx:
                    obs[gui.point_names[pt_idx]] = (obs_tuple[1], obs_tuple[2])
        if obs:
            result[img_idx] = obs

    return result


# ====================================================================
# Main
# ====================================================================

def main():
    import argparse
    p = argparse.ArgumentParser(
        description='SfM-based aircraft marker 3D reconstruction')
    p.add_argument('images', nargs='+', help='Image sequence')
    p.add_argument('--config', default='configs/experiment_config.yaml')
    p.add_argument('--point-names', nargs='+', required=True,
                   help='Marker point names')
    p.add_argument('--labels-dir', default='annotations/aircraft_2d',
                   help='Directory with existing _points.yaml files')
    p.add_argument('--output', '-o',
                   default='configs/aircraft_points_sfm.yaml')
    args = p.parse_args()

    os.makedirs('output', exist_ok=True)

    # Load config
    with open(args.config, encoding='utf-8') as f:
        exp = yaml.safe_load(f)
    cal = exp['calibration']
    K = np.array([[cal['fx'], 0, cal['cx']],
                   [0, cal['fy'], cal['cy']],
                   [0, 0, 1]], dtype=np.float64)
    dist = np.array(cal['dist'], dtype=np.float64)

    # Load images
    images, image_paths = [], []
    for pth in args.images:
        img = cv2.imread(pth)
        if img is not None:
            images.append(img)
            image_paths.append(pth)

    # Check for existing annotations
    annotation_files = []
    for pth in image_paths:
        stem = Path(pth).stem
        yaml_path = Path(args.labels_dir) / f'{stem}_points.yaml'
        if yaml_path.exists():
            annotation_files.append(str(yaml_path))
        else:
            annotation_files.append(None)  # will need manual labeling

    existing = [f for f in annotation_files if f is not None]
    if len(existing) >= 3:
        print(f'Found {len(existing)} existing annotation files')
    else:
        print(f'Only {len(existing)} annotation files, need >=3')

    # SfM reconstruction
    sfm = SfMReconstructor(K, dist)
    sfm.load_annotations(existing, args.point_names)

    if len(sfm.observations) < 2:
        print("Need at least 2 views with annotations. Run triangulation tool first.")
        sys.exit(1)

    # Board estimator for scale
    board_est = BoardPoseEstimator(K, dist)

    # Reconstruct
    points3d = sfm.reconstruct(image_paths, board_est)

    if not points3d:
        print("Reconstruction failed")
        sys.exit(1)

    # Export
    data = {
        'aircraft_name': 'model_jet',
        'coordinate_system': 'G',
        'unit': 'mm',
        'method': 'SfM (E-matrix decomposition + board scale)',
        'scale_mm': round(float(sfm.scale), 1),
        'point_count': len(points3d),
        'points': {}
    }
    for name, p in points3d.items():
        data['points'][name] = {
            'x_mm': round(float(p[0]), 2),
            'y_mm': round(float(p[1]), 2),
            'z_mm': round(float(p[2]), 2),
        }

    # Add Chinese aliases if point names are English
    cn_map = {
        'Cockpit': '机舱顶', 'L_Wingtip': '左翼尖', 'R_Wingtip': '右翼尖',
        'Spine': '机脊中部', 'L_HTail': '左横尾翼尖', 'R_HTail': '右横尾翼尖',
        'L_VTail': '左竖尾翼尖', 'R_VTail': '右竖尾翼尖',
    }
    cn_points = {}
    for en, cn in cn_map.items():
        if en in data['points']:
            cn_points[cn] = data['points'][en]
    if cn_points:
        data['points_chinese'] = cn_points

    with open(args.output, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False,
                  allow_unicode=True, sort_keys=False)
    print(f'\nSaved: {args.output}')

    # Summary
    z_vals = [info['z_mm'] for info in data['points'].values()]
    if len(z_vals) >= 2:
        z_range = max(z_vals) - min(z_vals)
        print(f'Points: {len(points3d)}, Z spread: {z_range:.1f} mm')
    for name, p in data['points'].items():
        print(f'  {name}: ({p["x_mm"]:.1f}, {p["y_mm"]:.1f}, {p["z_mm"]:.1f})')


if __name__ == '__main__':
    main()
