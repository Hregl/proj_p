"""
Convert aircraft marker points from board frame (G) to aircraft body frame (B).

Defines the B-frame coordinate system using reference points:
  - Origin: wing-root center (approximated from spine + wingtips)
  - X_B: tail -> nose
  - Y_B: left wing -> right wing
  - Z_B: toward belly (right-hand rule: Z = X x Y)

Usage:
  python tools/convert_to_B_frame.py
"""
import yaml, numpy as np, math

POINT_NAMES = {
    'nose': '机舱顶',
    'left_wing': '左翼尖',
    'right_wing': '右翼尖',
    'spine': '机脊中部',
    'left_tail': '左横尾翼尖',
    'right_tail': '右横尾翼尖',
    'left_vtail': '左竖尾翼尖',
    'right_vtail': '右竖尾翼尖',
}


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def build_B_frame(points_G: dict) -> tuple:
    """Build B-frame axes from G-frame reference points.

    Returns:
        (origin_G, x_B_G, y_B_G, z_B_G, G_R_B)
    """
    nose = np.array(points_G[POINT_NAMES['nose']])
    left = np.array(points_G[POINT_NAMES['left_wing']])
    right = np.array(points_G[POINT_NAMES['right_wing']])
    spine = np.array(points_G[POINT_NAMES['spine']])
    left_tail = np.array(points_G[POINT_NAMES['left_tail']])
    right_tail = np.array(points_G[POINT_NAMES['right_tail']])

    # --- Origin: wing-root center ---
    # Approximate as: spine projected to midpoint between wingtips in XZ plane,
    # with Y at the centerline (average of left and right Y at that X)
    tail_center = (left_tail + right_tail) / 2.0
    origin_G = spine.copy()

    # --- X_B: tail -> nose (unit vector) ---
    x_raw = nose - tail_center
    x_B_G = normalize(x_raw)

    # --- Y_B: left -> right, orthogonalized ---
    y_raw = right - left
    # Remove projection onto X
    y_ortho = y_raw - np.dot(y_raw, x_B_G) * x_B_G
    y_B_G = normalize(y_ortho)

    # --- Z_B = X x Y (points UP from board, away from belly) ---
    # B-frame is right-handed: X=nose, Y=right, Z=XxY=up.
    # The aircraft belly faces DOWN (-Z_G), which is -Z_B.
    # Document: 'z_axis: toward belly (belly = -Z_B direction)'
    z_B_G = np.cross(x_B_G, y_B_G)

    # Verify Y_B points left -> right.
    if np.dot(y_B_G, right - origin_G) < 0:
        y_B_G = -y_B_G; z_B_G = -z_B_G
        print("  Flipped Y (was pointing right -> left)")

    # Verify X_B points tail -> nose.
    if np.dot(x_B_G, nose - origin_G) < 0:
        x_B_G = -x_B_G; z_B_G = -z_B_G
        print("  Flipped X (was pointing nose -> tail)")

    # Build rotation matrix: G_R_B = [x_B_G | y_B_G | z_B_G]
    G_R_B = np.column_stack([x_B_G, y_B_G, z_B_G])

    # Enforce proper rotation (det=+1) via SVD
    det = np.linalg.det(G_R_B)
    if abs(det - 1.0) > 0.001:
        U, _, Vt = np.linalg.svd(G_R_B)
        G_R_B = U @ Vt
        if np.linalg.det(G_R_B) < 0:
            G_R_B[:, 2] *= -1
        x_B_G, y_B_G, z_B_G = G_R_B[:, 0], G_R_B[:, 1], G_R_B[:, 2]

    return origin_G, x_B_G, y_B_G, z_B_G, G_R_B


def convert_to_B(points_G: dict, origin_G: np.ndarray, G_R_B: np.ndarray) -> dict:
    """Convert all G-frame points to B-frame."""
    points_B = {}
    for name, coords in points_G.items():
        p_G = np.array(coords)
        p_B = G_R_B.T @ (p_G - origin_G)
        points_B[name] = {
            'x_mm': round(float(p_B[0]), 2),
            'y_mm': round(float(p_B[1]), 2),
            'z_mm': round(float(p_B[2]), 2),
        }
    return points_B


def main():
    print("=== Aircraft Body Frame (B) Construction ===\n")

    # Load G-frame points
    with open('configs/aircraft_points.yaml', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    points_G_raw = data['points']

    # Extract coordinate arrays
    points_G = {}
    for k in POINT_NAMES.values():
        if k in points_G_raw:
            points_G[k] = [points_G_raw[k]['x_mm'],
                           points_G_raw[k]['y_mm'],
                           points_G_raw[k]['z_mm']]

    available = [POINT_NAMES[k] for k in ['nose','left_wing','right_wing','spine','left_tail','right_tail']
                 if POINT_NAMES[k] in points_G]
    print(f"Reference points available: {len(available)}/6")
    for k, v in points_G.items():
        print(f"  {k}: ({v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f}) mm")

    # Build B-frame
    origin_G, x_B_G, y_B_G, z_B_G, G_R_B = build_B_frame(points_G)

    print(f"\nB-frame definition:")
    print(f"  Origin (G):  ({origin_G[0]:.1f}, {origin_G[1]:.1f}, {origin_G[2]:.1f}) mm")
    print(f"  X_B (G):     ({x_B_G[0]:.4f}, {x_B_G[1]:.4f}, {x_B_G[2]:.4f})  tail->nose")
    print(f"  Y_B (G):     ({y_B_G[0]:.4f}, {y_B_G[1]:.4f}, {y_B_G[2]:.4f})  left->right")
    print(f"  Z_B (G):     ({z_B_G[0]:.4f}, {z_B_G[1]:.4f}, {z_B_G[2]:.4f})  toward belly")
    print(f"  det(G_R_B):  {np.linalg.det(G_R_B):.6f}")

    # Orthogonality check
    dot_xy = np.dot(x_B_G, y_B_G)
    dot_xz = np.dot(x_B_G, z_B_G)
    dot_yz = np.dot(y_B_G, z_B_G)
    print(f"  Orthogonality: x.y={dot_xy:.6f}, x.z={dot_xz:.6f}, y.z={dot_yz:.6f}")

    # Convert all points to B-frame
    print(f"\n=== B-frame Points ===")
    points_B = convert_to_B(points_G, origin_G, G_R_B)

    z_vals = [v['z_mm'] for v in points_B.values()]
    print(f"  {'Name':<14} {'X_B(mm)':>9} {'Y_B(mm)':>9} {'Z_B(mm)':>9}")
    print(f"  {'-'*36}")
    for name in POINT_NAMES.values():
        if name in points_B:
            p = points_B[name]
            print(f"  {name:<14} {p['x_mm']:>9.1f} {p['y_mm']:>9.1f} {p['z_mm']:>9.1f}")

    if z_vals:
        print(f"\n  Z_B range: {min(z_vals):.1f} ~ {max(z_vals):.1f} mm (spread {max(z_vals)-min(z_vals):.1f} mm)")

    # --- Save B-frame file ---
    b_data = {
        'aircraft_name': 'model_jet',
        'coordinate_system': 'B',
        'unit': 'mm',
        'origin_definition': 'wing-root center (spine point)',
        'x_axis': 'tail -> nose',
        'y_axis': 'left wing -> right wing',
        'z_axis': 'toward belly (belly = -Z_B, Z_B = XxY points up)',
        'euler_convention': 'ZYX (yaw-pitch-roll)',
        'point_count': len(points_B),
        'note': ('Converted from G-frame triangulation. '
                 'B-frame origin at spine, X along fuselage axis, '
                 'Y across wingspan, Z toward belly.'),
        'G_R_B': [[round(float(v), 6) for v in row] for row in G_R_B.T],
        'origin_G_mm': [round(float(v), 1) for v in origin_G],
        'points': points_B,
    }
    with open('configs/aircraft_points_B.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(b_data, f, default_flow_style=False,
                  allow_unicode=True, sort_keys=False)
    print(f"\nSaved: configs/aircraft_points_B.yaml")

    # --- Keep G-frame as reference ---
    import shutil
    shutil.copy('configs/aircraft_points.yaml',
                'configs/aircraft_points_G_reference.yaml')
    print(f"Saved: configs/aircraft_points_G_reference.yaml")


if __name__ == '__main__':
    main()
