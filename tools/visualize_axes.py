"""
P1: Coordinate axis visualization module.

Projects board and aircraft coordinate axes onto an image for
visual verification of pose correctness.

Usage:
  python tools/visualize_axes.py data/planeNew/MVIMG_20260707_202357.jpg \
      --board-pose output/MVIMG_20260707_202357_board_pose.csv \
      --aircraft-pose output/MVIMG_20260707_202357_aircraft_pose.csv \
      --output output/axes_visualization.png
"""
import sys, yaml, cv2, numpy as np
from pathlib import Path

def load_csv_pose(filepath):
    with open(filepath) as f:
        for line in f:
            if line.startswith('image_id'): continue
            parts = line.strip().split(',')
            if len(parts) >= 7:
                rvec = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                tvec = np.array([float(parts[4]), float(parts[5]), float(parts[6])])
                R, _ = cv2.Rodrigues(rvec)
                return R, tvec.ravel(), rvec, float(parts[7]) if len(parts) > 7 else 0
    return None, None, None, 0

def draw_axis(img, K, dist, rvec, tvec, origin, length, color, label,
              thickness=2):
    """Draw 3 axes from origin with given length and color."""
    axis = np.array([[length, 0, 0],
                      [0, length, 0],
                      [0, 0, length]], dtype=np.float32)
    pts3d = np.array([origin, origin + axis[0], origin + axis[1], origin + axis[2]],
                     dtype=np.float32)
    pts2d, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist)
    pts2d = pts2d.reshape(-1, 2).astype(int)

    # Draw lines from origin
    labels = ['X', 'Y', 'Z']
    axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: R=X, G=Y, B=Z
    for i in range(3):
        cv2.line(img, tuple(pts2d[0]), tuple(pts2d[i+1]), axis_colors[i], thickness)
        cv2.putText(img, f"{label}{labels[i]}",
                   (pts2d[i+1][0] + 5, pts2d[i+1][1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, axis_colors[i], 2)
    cv2.circle(img, tuple(pts2d[0]), 5, color, -1)
    cv2.putText(img, label, (pts2d[0][0] + 8, pts2d[0][1] - 8),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img

def main():
    import argparse
    p = argparse.ArgumentParser(
        description='Project coordinate axes onto an image for pose verification')
    p.add_argument('image', help='Input image')
    p.add_argument('--config', default='configs/experiment_config.yaml')
    p.add_argument('--board-pose', required=True, help='Board PnP CSV')
    p.add_argument('--aircraft-pose', required=True, help='Aircraft PnP CSV')
    p.add_argument('--axis-length', type=float, default=50.0,
                   help='Axis length in mm')
    p.add_argument('--aircraft-3d', default='configs/aircraft_points.yaml',
                   help='Aircraft 3D points YAML (for drawing points)')
    p.add_argument('--output', '-o', default='output/axes_visualization.png')
    args = p.parse_args()

    with open(args.config, encoding='utf-8') as f: exp = yaml.safe_load(f)
    cal = exp['calibration']
    K = np.array([[cal['fx'], 0, cal['cx']],
                   [0, cal['fy'], cal['cy']],
                   [0, 0, 1]], dtype=np.float64)
    dist = np.array(cal['dist'], dtype=np.float64)

    # Load camera pose
    C_R_G, C_t_G, rvec_board, board_rmse = load_csv_pose(args.board_pose)
    if C_R_G is None:
        print("Invalid board pose CSV"); sys.exit(1)

    # Load aircraft pose
    C_R_B, C_t_B, rvec_ac, ac_rmse = load_csv_pose(args.aircraft_pose)
    if C_R_B is None:
        print("Invalid aircraft pose CSV"); sys.exit(1)

    img = cv2.imread(args.image)
    if img is None:
        print(f"Cannot read {args.image}"); sys.exit(1)

    L = args.axis_length

    # Draw board axes at origin of G frame
    img = draw_axis(img, K, dist, rvec_board, C_t_G,  # rvec/tvec for board→camera
                    np.array([0, 0, 0]), L, (0, 255, 255), 'G',
                    thickness=3)

    # Draw aircraft axes at origin of B frame
    # The board PnP gives C_T_G. We need the aircraft origin in C frame.
    # From aircraft PnP: C_T_B directly
    img = draw_axis(img, K, dist, rvec_ac, C_t_B,  # rvec/tvec for aircraft→camera
                    np.array([0, 0, 0]), L * 0.6, (255, 0, 255), 'B',
                    thickness=2)

    # Draw aircraft 3D points
    try:
        with open(args.aircraft_3d, encoding='utf-8') as f: ac3d = yaml.safe_load(f)
        pts_3d = []
        pts3d_all = ac3d.get('points', {})
        if ac3d.get('points_chinese'):
            pts3d_all = {**pts3d_all, **ac3d['points_chinese']}
        for name, info in pts3d_all.items():
            pts_3d.append([float(info['x_mm']), float(info['y_mm']), float(info['z_mm'])])
        if pts_3d:
            pts_arr = np.array(pts_3d, dtype=np.float32)
            proj, _ = cv2.projectPoints(pts_arr, rvec_ac, C_t_B, K, dist)
            for i, p in enumerate(proj.reshape(-1, 2).astype(int)):
                cv2.circle(img, tuple(p), 4, (255, 255, 0), -1)
    except Exception:
        pass

    # Legend
    legend = [
        (f"Board RMSE: {board_rmse:.2f}px", (0, 255, 255)),
        (f"Aircraft RMSE: {ac_rmse:.2f}px", (255, 0, 255)),
        ("G = Board frame (ref)", (0, 255, 255)),
        ("B = Aircraft body", (255, 0, 255)),
        ("X=red(R) Y=green(G) Z=blue(B)", (255, 255, 255)),
    ]
    y0 = 30
    for i, (text, color) in enumerate(legend):
        cv2.putText(img, text, (10, y0 + i * 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, img)
    print(f"Saved: {args.output}")

if __name__ == '__main__':
    main()
