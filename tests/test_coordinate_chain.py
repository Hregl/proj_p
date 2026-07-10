"""
Coordinate chain verification tests (V2 — uses canonical transforms).

Tests:
  A: Synthetic recovery — known camera + aircraft pose → PnP → compose
  B: Zero pose — B-frame aligned with G-frame
  C: Single-axis sign — +5 deg on each axis independently
  D: Non-identity camera rotation — realistic camera pose
  E: Random poses (100 groups) with pixel noise
  F: G→B→G point roundtrip
  Validation: G-frame rejection by estimate_aircraft_pose.py

All transforms imported from sls_calib/transforms.py.
"""
import sys, yaml, cv2, numpy as np, math, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sls_calib.transforms import (
    euler_to_R, R_to_euler, rotation_angle_error,
    compose_G_T_B, invert_pose, project_points, pnp, add_noise
)


# ====================================================================
# Config loading
# ====================================================================

def load_configs():
    with open('configs/cameras/camera_25mm_far.yaml', encoding='utf-8') as f:
        exp = yaml.safe_load(f)
    cal = exp['calibration']
    K = np.array([[cal['fx'], 0, cal['cx']],
                   [0, cal['fy'], cal['cy']],
                   [0, 0, 1]], dtype=np.float64)
    dist = np.array(cal['dist'], dtype=np.float64)

    with open('configs/aircraft_points_B.yaml', encoding='utf-8') as f:
        ac_b = yaml.safe_load(f)
    if ac_b.get('coordinate_system') != 'B':
        raise ValueError('B-frame points required')

    b_pts = {}
    for name, info in ac_b['points'].items():
        b_pts[name] = np.array([info['x_mm'], info['y_mm'], info['z_mm']],
                               dtype=np.float64)

    with open('configs/board_points.yaml', encoding='utf-8') as f:
        board = yaml.safe_load(f)
    g_pts = {}
    for name, info in board['points'].items():
        g_pts[name] = (np.array(info, dtype=np.float64) if isinstance(info, list)
                       else np.array([info['x_mm'], info['y_mm'], info['z_mm']],
                                     dtype=np.float64))

    return K, dist, b_pts, g_pts


def compose_and_report(C_R_G, C_t_G, C_R_B, C_t_B):
    """Compose G_T_B and return (yaw, pitch, roll, total_angle_error vs true)."""
    G_R_B, G_t_B = compose_G_T_B(C_R_G, C_t_G, C_R_B, C_t_B)
    yaw, pitch, roll = R_to_euler(G_R_B)
    return G_R_B, yaw, pitch, roll


# ====================================================================
# Test A: Synthetic recovery
# ====================================================================

def test_A():
    """Known angles → project → PnP → compose. Error should be 0."""
    print("=" * 60)
    print("Test A: Synthetic data recovery (identity camera)")
    print("=" * 60)

    K, dist, b_pts, g_pts = load_configs()

    yaw_t, pitch_t, roll_t = 10.0, 5.0, -3.0
    G_R_B_true = euler_to_R(yaw_t, pitch_t, roll_t)
    G_t_B_true = np.array([100.0, 50.0, -20.0])

    # Camera identity → board in front
    C_R_G = np.eye(3)
    C_t_G = np.array([0.0, 0.0, 500.0])

    # C_T_B = C_T_G * G_T_B
    C_R_B = C_R_G @ G_R_B_true
    C_t_B = C_R_G @ G_t_B_true + C_t_G

    b_2d = project_points(b_pts, K, dist, C_R_B, C_t_B)
    g_2d = project_points(g_pts, K, dist, C_R_G, C_t_G)

    C_R_G_est, C_t_G_est, g_rmse = pnp(g_2d, g_pts, K, dist)
    C_R_B_est, C_t_B_est, b_rmse = pnp(b_2d, b_pts, K, dist)

    if C_R_G_est is None or C_R_B_est is None:
        print("  FAIL: PnP failed"); return False

    G_R_B_est, yaw_e, pitch_e, roll_e = compose_and_report(
        C_R_G_est, C_t_G_est, C_R_B_est, C_t_B_est)

    yaw_err = abs(yaw_e - yaw_t)
    pitch_err = abs(pitch_e - pitch_t)
    roll_err = abs(roll_e - roll_t)
    angle_err = rotation_angle_error(G_R_B_true, G_R_B_est)

    print(f"  Board   PnP RMSE: {g_rmse:.4f} px")
    print(f"  Aircraft PnP RMSE: {b_rmse:.4f} px")
    print(f"  True:     yaw={yaw_t:.4f}  pitch={pitch_t:.4f}  roll={roll_t:.4f}")
    print(f"  Recovered: yaw={yaw_e:.4f}  pitch={pitch_e:.4f}  roll={roll_e:.4f}")
    print(f"  Euler err: yaw={yaw_err:.4f}  pitch={pitch_err:.4f}  roll={roll_err:.4f}")
    print(f"  Total angle error: {angle_err:.4f} deg")

    passed = max(yaw_err, pitch_err, roll_err) < 0.01 and angle_err < 0.01
    print(f"  {'PASSED' if passed else 'FAILED'}")
    return passed


# ====================================================================
# Test B: Zero pose
# ====================================================================

def test_B():
    print("\n" + "=" * 60)
    print("Test B: Zero pose (identity camera)")
    print("=" * 60)

    K, dist, b_pts, g_pts = load_configs()
    G_R_B_true = np.eye(3)
    C_R_G = np.eye(3)
    C_t_G = np.array([0.0, 0.0, 500.0])
    C_R_B = C_R_G @ G_R_B_true
    C_t_B = C_R_G @ np.zeros(3) + C_t_G

    b_2d = project_points(b_pts, K, dist, C_R_B, C_t_B)
    g_2d = project_points(g_pts, K, dist, C_R_G, C_t_G)

    C_R_G_est, C_t_G_est, _ = pnp(g_2d, g_pts, K, dist)
    C_R_B_est, C_t_B_est, _ = pnp(b_2d, b_pts, K, dist)
    if C_R_G_est is None or C_R_B_est is None:
        print("  FAIL: PnP failed"); return False

    _, yaw, pitch, roll = compose_and_report(
        C_R_G_est, C_t_G_est, C_R_B_est, C_t_B_est)
    print(f"  Recovered: yaw={yaw:.4f}  pitch={pitch:.4f}  roll={roll:.4f}")
    passed = max(abs(yaw), abs(pitch), abs(roll)) < 0.01
    print(f"  {'PASSED' if passed else 'FAILED'}")
    return passed


# ====================================================================
# Test C: Single-axis sign
# ====================================================================

def test_C():
    print("\n" + "=" * 60)
    print("Test C: Single-axis sign (identity camera)")
    print("=" * 60)

    K, dist, b_pts, g_pts = load_configs()
    C_R_G = np.eye(3)
    C_t_G = np.array([0.0, 0.0, 500.0])
    all_ok = True

    for label, y, p, r in [('yaw +5', 5, 0, 0),
                            ('pitch +5', 0, 5, 0),
                            ('roll +5', 0, 0, 5)]:
        G_R_B = euler_to_R(y, p, r)
        C_R_B = C_R_G @ G_R_B
        C_t_B = C_t_G
        b_2d = project_points(b_pts, K, dist, C_R_B, C_t_B)
        g_2d = project_points(g_pts, K, dist, C_R_G, C_t_G)

        C_R_G_est, C_t_G_est, _ = pnp(g_2d, g_pts, K, dist)
        C_R_B_est, C_t_B_est, _ = pnp(b_2d, b_pts, K, dist)
        if C_R_G_est is None or C_R_B_est is None:
            print(f"  {label}: FAIL (PnP)"); all_ok = False; continue

        _, yaw, pitch, roll = compose_and_report(
            C_R_G_est, C_t_G_est, C_R_B_est, C_t_B_est)
        errs = [abs(yaw-y), abs(pitch-p), abs(roll-r)]
        ok = max(errs) < 0.01
        print(f"  {label}: yaw={yaw:+.4f} pitch={pitch:+.4f} roll={roll:+.4f}  "
              f"errs={[f'{e:.4f}' for e in errs]} {'OK' if ok else 'FAIL'}")
        if not ok: all_ok = False

    print(f"  {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    return all_ok


# ====================================================================
# Test D: Non-identity camera rotation (NEW — catches the bug!)
# ====================================================================

def test_D():
    """Realistic camera pose: board rotated 23/-17/8 deg."""
    print("\n" + "=" * 60)
    print("Test D: Non-identity camera rotation (camera yaw=23 pitch=-17 roll=8)")
    print("=" * 60)

    K, dist, b_pts, g_pts = load_configs()

    # Aircraft pose in board frame
    yaw_t, pitch_t, roll_t = 10.0, 5.0, -3.0
    G_R_B_true = euler_to_R(yaw_t, pitch_t, roll_t)
    G_t_B_true = np.array([100.0, 50.0, -20.0])

    # Camera pose: rotated relative to board
    C_R_G = euler_to_R(23.0, -17.0, 8.0)
    C_t_G = np.array([150.0, -80.0, 600.0])

    # Project
    C_R_B = C_R_G @ G_R_B_true
    C_t_B = C_R_G @ G_t_B_true + C_t_G
    b_2d = project_points(b_pts, K, dist, C_R_B, C_t_B)
    g_2d = project_points(g_pts, K, dist, C_R_G, C_t_G)

    # PnP recovery
    C_R_G_est, C_t_G_est, _ = pnp(g_2d, g_pts, K, dist)
    C_R_B_est, C_t_B_est, _ = pnp(b_2d, b_pts, K, dist)
    if C_R_G_est is None or C_R_B_est is None:
        print("  FAIL: PnP failed"); return False

    # Compose
    G_R_B_est, yaw_e, pitch_e, roll_e = compose_and_report(
        C_R_G_est, C_t_G_est, C_R_B_est, C_t_B_est)

    angle_err = rotation_angle_error(G_R_B_true, G_R_B_est)

    print(f"  True:     yaw={yaw_t:.4f}  pitch={pitch_t:.4f}  roll={roll_t:.4f}")
    print(f"  Recovered: yaw={yaw_e:.4f}  pitch={pitch_e:.4f}  roll={roll_e:.4f}")
    print(f"  Total angle error: {angle_err:.4f} deg")

    euler_ok = max(abs(yaw_e-yaw_t), abs(pitch_e-pitch_t), abs(roll_e-roll_t)) < 0.01
    angle_ok = angle_err < 0.01
    passed = euler_ok and angle_ok
    print(f"  Euler {'OK' if euler_ok else 'FAIL'}, Angle {'OK' if angle_ok else 'FAIL'}")
    print(f"  {'PASSED' if passed else 'FAILED'}")
    return passed


# ====================================================================
# Test E: Random poses + pixel noise
# ====================================================================

def test_E():
    """100 random camera + aircraft poses with 0.1/0.3/0.5 px noise."""
    print("\n" + "=" * 60)
    print("Test E: Random poses (100 groups) with pixel noise")
    print("=" * 60)

    K, dist, b_pts, g_pts = load_configs()
    rng = np.random.default_rng(42)

    results = {}
    for sigma in [0.1, 0.3, 0.5]:
        angle_errs = []
        for i in range(100):
            # Random camera pose
            cam_y, cam_p, cam_r = rng.uniform(-30, 30, 3)
            C_R_G = euler_to_R(float(cam_y), float(cam_p), float(cam_r))
            C_t_G = np.array([rng.uniform(-200, 200),
                              rng.uniform(-200, 200),
                              rng.uniform(400, 800)])

            # Random aircraft pose
            ac_y, ac_p, ac_r = rng.uniform(-15, 15, 3)
            G_R_B_true = euler_to_R(float(ac_y), float(ac_p), float(ac_r))

            # Project
            C_R_B = C_R_G @ G_R_B_true
            C_t_B = C_R_G @ np.zeros(3) + C_t_G
            b_2d_clean = project_points(b_pts, K, dist, C_R_B, C_t_B)
            g_2d_clean = project_points(g_pts, K, dist, C_R_G, C_t_G)

            # Add noise
            b_2d_noisy = add_noise(b_2d_clean, sigma, seed=i)
            g_2d_noisy = add_noise(g_2d_clean, sigma, seed=i+1000)

            # PnP recovery
            C_R_G_est, C_t_G_est, _ = pnp(g_2d_noisy, g_pts, K, dist)
            C_R_B_est, C_t_B_est, _ = pnp(b_2d_noisy, b_pts, K, dist)
            if C_R_G_est is None or C_R_B_est is None:
                continue

            G_R_B_est, _, _, _ = compose_and_report(
                C_R_G_est, C_t_G_est, C_R_B_est, C_t_B_est)
            angle_errs.append(rotation_angle_error(G_R_B_true, G_R_B_est))

        if angle_errs:
            arr = np.array(angle_errs)
            results[sigma] = {
                'n': len(arr), 'mean': np.mean(arr), 'std': np.std(arr),
                'max': np.max(arr), 'p95': np.percentile(arr, 95)
            }

    for sigma in sorted(results):
        r = results[sigma]
        print(f"  sigma={sigma:.1f}px: n={r['n']:>3}, mean={r['mean']:.4f} deg, "
              f"std={r['std']:.4f} deg, p95={r['p95']:.4f} deg, max={r['max']:.4f} deg")

    # With sigma=0.1, mean angle error should be < 0.02 deg (~1 arcmin noise floor)
    # With sigma=0.5, mean angle error should be < 0.10 deg (~6 arcmin, realistic worst)
    passed = (results.get(0.1, {}).get('mean', 999) < 0.02 and
              results.get(0.5, {}).get('mean', 999) < 0.10)
    print(f"  {'PASSED' if passed else 'FAILED'} "
          f"(sigma=0.1 mean < 0.02 deg, sigma=0.5 mean < 0.10 deg)")
    return passed


# ====================================================================
# Test F: G→B→G point roundtrip
# ====================================================================

def test_F():
    """Convert points G→B→G via transforms, check roundtrip error."""
    print("\n" + "=" * 60)
    print("Test F: G->B->G point coordinate roundtrip")
    print("=" * 60)

    K, dist, b_pts, g_pts = load_configs()

    # Get G→B transform from the B-frame config
    with open('configs/aircraft_points_B.yaml', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    G_R_B_cfg = np.array(cfg.get('G_R_B', [[1,0,0],[0,1,0],[0,0,1]]))
    origin_G = np.array(cfg.get('origin_G_mm', [0, 0, 0]))

    # Test: project G-frame points through the transform chain
    # G point → B point = G_R_B^T * (P_G - origin)
    max_err = 0
    for name in g_pts:
        if name not in cfg.get('points', {}):
            continue
        p_G = g_pts[name]
        p_B_expected = np.array([
            cfg['points'][name]['x_mm'],
            cfg['points'][name]['y_mm'],
            cfg['points'][name]['z_mm']])
        # G→B conversion
        p_B_computed = G_R_B_cfg.T @ (p_G - origin_G)
        err = float(np.linalg.norm(p_B_computed - p_B_expected))
        if err > max_err:
            max_err = err
        if err > 0.5:
            print(f"  {name}: roundtrip error = {err:.3f} mm")

    print(f"  Max roundtrip error: {max_err:.4f} mm")
    passed = max_err < 0.5
    print(f"  {'PASSED' if passed else 'FAILED'} (threshold < 0.5 mm)")
    return passed


# ====================================================================
# Validation test
# ====================================================================

def test_validation():
    """Verify G-frame rejection and B-frame acceptance."""
    print("\n" + "=" * 60)
    print("Test: G-frame rejection / B-frame acceptance")
    print("=" * 60)

    py = sys.executable

    r = subprocess.run(
        [py, 'tools/estimate_aircraft_pose.py',
         '--config', 'configs/cameras/camera_25mm_far.yaml',
         '--aircraft-3d', 'configs/aircraft_points_G_reference.yaml',
         '--aircraft-2d',
         'annotations/aircraft_2d/Pic_2026_07_09_193004_131_points.yaml',
         '-o', 'output/test_g_reject.csv'],
        capture_output=True, text=True)
    g_ok = 'aircraft body frame (B)' in (r.stdout + r.stderr)

    r2 = subprocess.run(
        [py, 'tools/estimate_aircraft_pose.py',
         '--config', 'configs/cameras/camera_25mm_far.yaml',
         '--aircraft-3d', 'configs/aircraft_points_B.yaml',
         '--aircraft-2d',
         'annotations/aircraft_2d/Pic_2026_07_09_193004_131_points.yaml',
         '-o', 'output/test_b_accept.csv'],
        capture_output=True, text=True)
    b_ok = 'inliers' in r2.stdout

    for f in ['output/test_g_reject.csv', 'output/test_b_accept.csv']:
        Path(f).unlink(missing_ok=True)

    print(f"  G-frame rejected: {'PASS' if g_ok else 'FAIL'}")
    print(f"  B-frame accepted: {'PASS' if b_ok else 'FAIL'}")
    return g_ok and b_ok


# ====================================================================
# Main
# ====================================================================

if __name__ == '__main__':
    tests = [
        ('A: synthetic (identity cam)', test_A),
        ('B: zero pose', test_B),
        ('C: single-axis sign', test_C),
        ('D: non-identity camera rotation', test_D),
        ('E: random poses + noise', test_E),
        ('F: G->B->G roundtrip', test_F),
        ('Validation', test_validation),
    ]

    results = {}
    for name, fn in tests:
        results[name] = fn()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_ok = True
    for name, ok in results.items():
        print(f"  {name}: {'PASSED' if ok else 'FAILED'}")
        if not ok: all_ok = False
    print(f"\n  Overall: {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    sys.exit(0 if all_ok else 1)
