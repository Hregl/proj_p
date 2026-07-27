"""
Stage 3: Convert aircraft marker points from ground frame (G) to body frame (B).

Defines the B-frame coordinate system using reference points:
  - Origin: wing-root center (approximated from spine + wingtips)
  - X_B: tail -> nose
  - Y_B: left wing -> right wing
  - Z_B: X x Y (points upward, away from belly)

Outputs aircraft_points_B.yaml with G_T_B transform, plus a
structured quality report (stage_03_B_frame_report.json).

Usage:
  python tools/convert_to_B_frame.py \
      --input configs/aircraft_points_G.yaml \
      --output configs/aircraft_points_B.yaml \
      --report output/session/stage_03_B_frame_report.json
"""

import yaml, numpy as np, math, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


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
        where G_R_B = [x_B_G | y_B_G | z_B_G] maps B→G.
    """
    nose = np.array(points_G[POINT_NAMES['nose']])
    left = np.array(points_G[POINT_NAMES['left_wing']])
    right = np.array(points_G[POINT_NAMES['right_wing']])
    spine = np.array(points_G[POINT_NAMES['spine']])
    left_tail = np.array(points_G[POINT_NAMES['left_tail']])
    right_tail = np.array(points_G[POINT_NAMES['right_tail']])

    tail_center = (left_tail + right_tail) / 2.0
    origin_G = spine.copy()

    # X_B: tail -> nose
    x_raw = nose - tail_center
    x_B_G = normalize(x_raw)

    # Y_B: left -> right, orthogonalized
    y_raw = right - left
    y_ortho = y_raw - np.dot(y_raw, x_B_G) * x_B_G
    y_B_G = normalize(y_ortho)

    # Z_B = X x Y (right-handed)
    z_B_G = np.cross(x_B_G, y_B_G)

    # Direction checks
    if np.dot(y_B_G, right - origin_G) < 0:
        y_B_G = -y_B_G; z_B_G = -z_B_G
        print("  Flipped Y (was pointing right -> left)")

    if np.dot(x_B_G, nose - origin_G) < 0:
        x_B_G = -x_B_G; z_B_G = -z_B_G
        print("  Flipped X (was pointing nose -> tail)")

    # Build rotation matrix
    G_R_B = np.column_stack([x_B_G, y_B_G, z_B_G])
    det_before = np.linalg.det(G_R_B)

    # Enforce proper rotation (det=+1) via SVD
    if abs(det_before - 1.0) > 0.001:
        U, _, Vt = np.linalg.svd(G_R_B)
        G_R_B = U @ Vt
        if np.linalg.det(G_R_B) < 0:
            G_R_B[:, 2] *= -1
        x_B_G, y_B_G, z_B_G = G_R_B[:, 0], G_R_B[:, 1], G_R_B[:, 2]

    return origin_G, x_B_G, y_B_G, z_B_G, G_R_B, det_before


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
    import argparse
    p = argparse.ArgumentParser(
        description='Stage 3: Convert aircraft points from G-frame to B-frame')
    p.add_argument('--input', default='configs/aircraft_points_G.yaml',
                   help='Input G-frame points YAML')
    p.add_argument('--output', default='configs/aircraft_points_B.yaml',
                   help='Output B-frame points YAML')
    p.add_argument('--report', default=None,
                   help='Output quality report JSON')
    args = p.parse_args()

    print("=== Aircraft Body Frame (B) Construction ===\n")

    # Load G-frame points
    with open(args.input, encoding='utf-8') as f:
        data = yaml.safe_load(f)

    cs = data.get('coordinate_system', '')
    if cs != 'G':
        raise ValueError(
            f"Input points must be in G frame, got coordinate_system={cs!r}. "
            f"File: {args.input}")

    points_G_raw = data['points']

    # Extract coordinate arrays
    points_G = {}
    missing_refs = []
    for key, name in POINT_NAMES.items():
        if name in points_G_raw:
            points_G[name] = [points_G_raw[name]['x_mm'],
                             points_G_raw[name]['y_mm'],
                             points_G_raw[name]['z_mm']]
        elif key in ['nose', 'left_wing', 'right_wing', 'spine',
                     'left_tail', 'right_tail']:
            missing_refs.append(name)

    if missing_refs:
        raise ValueError(
            f"Missing required reference points: {missing_refs}. "
            f"Available points: {list(points_G_raw.keys())}")

    print(f"Reference points available: {len(points_G)}")
    for k, v in points_G.items():
        print(f"  {k}: ({v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f}) mm")

    # Build B-frame
    origin_G, x_B_G, y_B_G, z_B_G, G_R_B, det_before = build_B_frame(points_G)

    det_after = np.linalg.det(G_R_B)
    dot_xy = float(np.dot(x_B_G, y_B_G))
    dot_xz = float(np.dot(x_B_G, z_B_G))
    dot_yz = float(np.dot(y_B_G, z_B_G))
    max_ortho = max(abs(dot_xy), abs(dot_xz), abs(dot_yz))

    print(f"\nB-frame definition:")
    print(f"  Origin (G):  ({origin_G[0]:.1f}, {origin_G[1]:.1f}, "
          f"{origin_G[2]:.1f}) mm")
    print(f"  X_B (G):     ({x_B_G[0]:.4f}, {x_B_G[1]:.4f}, {x_B_G[2]:.4f})")
    print(f"  Y_B (G):     ({y_B_G[0]:.4f}, {y_B_G[1]:.4f}, {y_B_G[2]:.4f})")
    print(f"  Z_B (G):     ({z_B_G[0]:.4f}, {z_B_G[1]:.4f}, {z_B_G[2]:.4f})")
    print(f"  det(G_R_B) before SVD: {det_before:.6f}")
    print(f"  det(G_R_B) after SVD:  {det_after:.6f} "
          f"({'OK' if abs(det_after - 1.0) < 0.001 else 'FAIL'})")
    print(f"  Orthogonality: x·y={dot_xy:.6f}, x·z={dot_xz:.6f}, "
          f"y·z={dot_yz:.6f} ({'OK' if max_ortho < 0.001 else 'WARN'})")

    # Convert all points to B-frame
    print(f"\n=== B-frame Points ===")
    points_B = convert_to_B(points_G, origin_G, G_R_B)

    z_vals = [v['z_mm'] for v in points_B.values()]
    print(f"  {'Name':<14} {'X_B(mm)':>9} {'Y_B(mm)':>9} {'Z_B(mm)':>9}")
    print(f"  {'-'*36}")
    for name, p in points_B.items():
        print(f"  {name:<14} {p['x_mm']:>9.1f} {p['y_mm']:>9.1f} "
              f"{p['z_mm']:>9.1f}")

    if z_vals:
        print(f"\n  Z_B range: {min(z_vals):.1f} ~ {max(z_vals):.1f} mm "
              f"(spread {max(z_vals)-min(z_vals):.1f} mm)")

    # --- G → B → G roundtrip check ---
    max_rt_err = 0.0
    rt_errors = {}
    for name, p_G in points_G.items():
        p_B_arr = np.array([points_B[name]['x_mm'],
                           points_B[name]['y_mm'],
                           points_B[name]['z_mm']])
        p_G_rt = G_R_B @ p_B_arr + origin_G
        err = float(np.linalg.norm(p_G_rt - p_G))
        rt_errors[name] = round(err, 6)
        if err > max_rt_err:
            max_rt_err = err
    print(f"\n  G→B→G roundtrip max error: {max_rt_err:.4f} mm "
          f"({'OK' if max_rt_err < 0.01 else 'FAIL'})")

    # --- Key structure distance checks ---
    structure_dists = {}
    structure_pairs = [
        ('nose_left_wing', '机舱顶', '左翼尖'),
        ('nose_right_wing', '机舱顶', '右翼尖'),
        ('left_right_wing', '左翼尖', '右翼尖'),
        ('nose_tail_center', '机舱顶', 'tail_center'),
    ]
    tail_center = (np.array(points_G['左横尾翼尖']) +
                   np.array(points_G['右横尾翼尖'])) / 2.0
    for label, a_name, b_name in structure_pairs:
        if a_name == 'tail_center':
            a_pt = tail_center
        else:
            a_pt = np.array(points_G[a_name])
        if b_name == 'tail_center':
            b_pt = tail_center
        else:
            b_pt = np.array(points_G[b_name])
        d = float(np.linalg.norm(a_pt - b_pt))
        structure_dists[label] = round(d, 1)
        print(f"  Structure {label}: {d:.1f} mm")

    # --- Save B-frame file ---
    b_data = {
        'aircraft_name': 'model_jet',
        'coordinate_system': 'B',
        'unit': 'mm',
        'origin_definition': 'wing-root center (spine point)',
        'x_axis': 'tail -> nose',
        'y_axis': 'left wing -> right wing',
        'z_axis': 'Z_B = X_B x Y_B points upward (away from belly). '
                  'Belly = -Z_B.',
        'euler_convention': 'ZYX (yaw-pitch-roll)',
        'point_count': len(points_B),
        'generated_at': __import__('time').strftime('%Y-%m-%dT%H:%M:%S'),
        'source_G_file': args.input,
        'note': ('Converted from G-frame triangulation. '
                 'B-frame origin at spine, X along fuselage axis, '
                 'Y across wingspan, Z toward belly.'),
        'G_R_B': [[round(float(v), 6) for v in row] for row in G_R_B],
        'origin_G_mm': [round(float(v), 1) for v in origin_G],
        'points': points_B,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        yaml.dump(b_data, f, default_flow_style=False,
                  allow_unicode=True, sort_keys=False)
    print(f"\nSaved: {args.output}")

    # --- Save report ---
    report_path = args.report or str(
        Path(args.output).with_suffix('') + '_report.json')
    report = {
        'stage': 'B_frame_construction',
        'status': 'pass' if (abs(det_after - 1.0) < 0.001 and
                            max_ortho < 0.001 and
                            max_rt_err < 0.01) else 'warning',
        'inputs': {'G_points_file': args.input},
        'outputs': {'B_points_file': args.output},
        'metrics': {
            'det_G_R_B': round(float(det_after), 6),
            'det_G_R_B_before_svd': round(float(det_before), 6),
            'orthogonality_x_dot_y': round(dot_xy, 6),
            'orthogonality_x_dot_z': round(dot_xz, 6),
            'orthogonality_y_dot_z': round(dot_yz, 6),
            'roundtrip_max_error_mm': round(max_rt_err, 6),
            'n_points_total': len(points_B),
        },
        'structure_distances_mm': structure_dists,
        'roundtrip_errors_mm': rt_errors,
        'warnings': [],
    }
    if abs(det_after - 1.0) >= 0.001:
        report['warnings'].append(f'det(G_R_B)={det_after:.6f} != 1.0')
    if max_ortho >= 0.001:
        report['warnings'].append(f'Max orthogonality error={max_ortho:.6f}')
    if max_rt_err >= 0.01:
        report['warnings'].append(f'Roundtrip error={max_rt_err:.4f} mm')

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report: {report_path}")


if __name__ == '__main__':
    main()
