"""
P1: Coordinate axis visualization module (Ground + Aircraft).

Projects ground (G) and aircraft body (B) coordinate axes, ground
marker 3D positions, and aircraft marker 3D positions onto an image
for visual pose verification.

Usage:
  python tools/visualize_axes.py data/planeNew/MVIMG_20260707_202357.jpg \
      --ground-pose output/MVIMG_20260707_202357_ground_pose.csv \
      --aircraft-pose output/MVIMG_20260707_202357_aircraft_pose.csv \
      --ground-3d configs/ground_markers_G.yaml \
      --aircraft-3d configs/aircraft_points_B.yaml \
      --output output/axes_visualization.png
"""
import sys, yaml, cv2, numpy as np
from pathlib import Path


def load_csv_pose(filepath):
    """Load rvec/tvec/rmse from a PnP CSV."""
    with open(filepath) as f:
        for line in f:
            if line.startswith('image_id'):
                continue
            parts = line.strip().split(',')
            if len(parts) >= 7:
                try:
                    rvec = np.array([float(parts[1]), float(parts[2]),
                                     float(parts[3])])
                    tvec = np.array([float(parts[4]), float(parts[5]),
                                     float(parts[6])])
                    R, _ = cv2.Rodrigues(rvec)
                    rmse = float(parts[7]) if len(parts) > 7 else 0
                    return R, tvec.ravel(), rvec, rmse
                except (ValueError, cv2.error):
                    continue
    return None, None, None, 0


def draw_axis(img, K, dist, rvec, tvec, origin, length, color, label,
              thickness=2):
    """Draw 3 axes from origin with given length and color."""
    axis = np.array([[length, 0, 0],
                     [0, length, 0],
                     [0, 0, length]], dtype=np.float32)
    pts3d = np.array([origin,
                      origin + axis[0],
                      origin + axis[1],
                      origin + axis[2]], dtype=np.float32)
    pts2d, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist)
    pts2d = pts2d.reshape(-1, 2).astype(int)

    axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: R=X, G=Y, B=Z
    for i in range(3):
        cv2.line(img, tuple(pts2d[0]), tuple(pts2d[i + 1]),
                 axis_colors[i], thickness)
        cv2.putText(img, f"{label}{['X','Y','Z'][i]}",
                    (pts2d[i + 1][0] + 5, pts2d[i + 1][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, axis_colors[i], 2)
    cv2.circle(img, tuple(pts2d[0]), 5, color, -1)
    cv2.putText(img, label, (pts2d[0][0] + 8, pts2d[0][1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img


def main():
    import argparse
    p = argparse.ArgumentParser(
        description='Project G and B coordinate axes onto an image for pose '
                    'verification')
    p.add_argument('image', help='Input image')
    p.add_argument('--config', default='configs/experiment_config.yaml')
    p.add_argument('--ground-pose', required=True,
                   help='Ground PnP CSV (C_T_G)')
    p.add_argument('--aircraft-pose', required=True,
                   help='Aircraft PnP CSV (C_T_B)')
    p.add_argument('--axis-length', type=float, default=50.0,
                   help='Axis length in mm')
    p.add_argument('--ground-3d', default='configs/ground_markers_G.yaml',
                   help='Ground marker 3D points YAML')
    p.add_argument('--aircraft-3d', default='configs/aircraft_points_B.yaml',
                   help='Aircraft 3D points YAML (B-frame)')
    p.add_argument('--output', '-o', default='output/axes_visualization.png')
    args = p.parse_args()

    with open(args.config, encoding='utf-8') as f:
        exp = yaml.safe_load(f)
    cal = exp['calibration']
    K = np.array([[cal['fx'], 0, cal['cx']],
                  [0, cal['fy'], cal['cy']],
                  [0, 0, 1]], dtype=np.float64)
    dist = np.array(cal['dist'], dtype=np.float64)

    # --- Load ground pose: C_T_G ---
    C_R_G, C_t_G, rvec_ground, ground_rmse = load_csv_pose(args.ground_pose)
    if C_R_G is None:
        print("Invalid ground pose CSV"); sys.exit(1)

    # --- Load aircraft pose: C_T_B ---
    C_R_B, C_t_B, rvec_ac, ac_rmse = load_csv_pose(args.aircraft_pose)
    if C_R_B is None:
        print("Invalid aircraft pose CSV"); sys.exit(1)

    img = cv2.imread(args.image)
    if img is None:
        print(f"Cannot read {args.image}"); sys.exit(1)

    L = args.axis_length

    # --- Draw Ground (G) axes at G origin ---
    img = draw_axis(img, K, dist, rvec_ground, C_t_G,
                    np.array([0, 0, 0]), L, (0, 255, 255), 'G_',
                    thickness=3)

    # --- Draw Ground markers (project 3D G points) ---
    try:
        with open(args.ground_3d, encoding='utf-8') as f:
            g3d = yaml.safe_load(f)
        g_pts = []
        g_labels = []
        for name, coords in g3d.get('points', {}).items():
            g_pts.append([float(coords[0]), float(coords[1]), float(coords[2])])
            g_labels.append(name)
        if g_pts:
            g_arr = np.array(g_pts, dtype=np.float32)
            proj_g, _ = cv2.projectPoints(g_arr, rvec_ground, C_t_G, K, dist)
            for i, pt in enumerate(proj_g.reshape(-1, 2).astype(int)):
                cv2.circle(img, tuple(pt), 5, (0, 255, 255), -1)
                cv2.circle(img, tuple(pt), 7, (0, 200, 200), 1)
                cv2.putText(img, g_labels[i],
                           (pt[0] + 8, pt[1] - 6),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    except Exception as e:
        print(f"Ground markers not drawn: {e}")

    # --- Draw Aircraft (B) axes at B origin ---
    img = draw_axis(img, K, dist, rvec_ac, C_t_B,
                    np.array([0, 0, 0]), L * 0.6, (255, 0, 255), 'B_',
                    thickness=2)

    # --- Draw Aircraft 3D points ---
    try:
        with open(args.aircraft_3d, encoding='utf-8') as f:
            ac3d = yaml.safe_load(f)
        pts_3d = []
        pts_labels = []
        pts3d_all = ac3d.get('points', {})
        if ac3d.get('points_chinese'):
            pts3d_all = {**pts3d_all, **ac3d['points_chinese']}
        for name, info in pts3d_all.items():
            pts_3d.append([float(info['x_mm']), float(info['y_mm']),
                          float(info['z_mm'])])
            pts_labels.append(name)
        if pts_3d:
            pts_arr = np.array(pts_3d, dtype=np.float32)
            proj, _ = cv2.projectPoints(pts_arr, rvec_ac, C_t_B, K, dist)
            for i, pt in enumerate(proj.reshape(-1, 2).astype(int)):
                cv2.circle(img, tuple(pt), 4, (255, 255, 0), -1)
                cv2.putText(img, pts_labels[i],
                           (pt[0] + 6, pt[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
    except Exception:
        pass

    # --- Legend ---
    legend = [
        (f"Ground RMSE: {ground_rmse:.2f}px", (0, 255, 255)),
        (f"Aircraft RMSE: {ac_rmse:.2f}px", (255, 0, 255)),
        ("G = Ground frame (measured markers)", (0, 255, 255)),
        ("B = Aircraft body frame", (255, 0, 255)),
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
