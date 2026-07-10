"""
Coordinate chain verification tests.

Test A: Synthetic data — known angles → project → recover (< 0.01 deg error)
Test B: Zero pose — B-frame aligned with G-frame (yaw≈pitch≈roll≈0)
Test C: Single-axis sign — +5 deg on each axis independently

Pass criteria:
  - B-frame point library generated (configs/aircraft_points_B.yaml)
  - estimate_aircraft_pose.py rejects G-frame points
  - All 3 tests pass with < 0.01 deg error
"""
import sys, yaml, cv2, numpy as np, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ====================================================================
# Utilities
# ====================================================================

def euler_to_R(yaw_deg, pitch_deg, roll_deg):
    """ZYX Euler → rotation matrix."""
    y, p, r = np.radians(yaw_deg), np.radians(pitch_deg), np.radians(roll_deg)
    Rz = np.array([[math.cos(y), -math.sin(y), 0],
                    [math.sin(y),  math.cos(y), 0],
                    [0, 0, 1]])
    Ry = np.array([[ math.cos(p), 0, math.sin(p)],
                    [0, 1, 0],
                    [-math.sin(p), 0, math.cos(p)]])
    Rx = np.array([[1, 0, 0],
                    [0, math.cos(r), -math.sin(r)],
                    [0, math.sin(r),  math.cos(r)]])
    return Rz @ Ry @ Rx


def R_to_euler(R):
    """Rotation matrix → ZYX Euler angles (degrees)."""
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        rx = math.atan2(R[2, 1], R[2, 2])
        ry = math.atan2(-R[2, 0], sy)
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        rx = math.atan2(-R[1, 2], R[1, 1])
        ry = math.atan2(-R[2, 0], sy)
        rz = 0.0
    return np.degrees(rx), np.degrees(ry), np.degrees(rz)


def load_configs():
    """Load K, dist, board points, B-frame points."""
    with open('configs/camera_20mm_far.yaml', encoding='utf-8') as f:
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

    with open('configs/board_points.yaml', encoding='utf-8') as f:
        board = yaml.safe_load(f)

    # Extract point arrays
    b_pts_3d = {}
    for name, info in ac_b['points'].items():
        b_pts_3d[name] = np.array([info['x_mm'], info['y_mm'], info['z_mm']],
                                  dtype=np.float64)

    g_pts_3d = {}
    for name, info in board['points'].items():
        if isinstance(info, list):
            g_pts_3d[name] = np.array(info, dtype=np.float64)
        else:
            g_pts_3d[name] = np.array([info['x_mm'], info['y_mm'], info['z_mm']],
                                      dtype=np.float64)

    return K, dist, b_pts_3d, g_pts_3d


def project_points(points_3d, K, dist, R, t):
    """Project 3D points to 2D image coordinates. Returns {name: (u,v)}."""
    pts = np.array(list(points_3d.values()), dtype=np.float64)
    rvec, _ = cv2.Rodrigues(R)
    proj, _ = cv2.projectPoints(pts, rvec, t, K, dist)
    result = {}
    for i, name in enumerate(points_3d.keys()):
        result[name] = (float(proj[i, 0, 0]), float(proj[i, 0, 1]))
    return result


def pnp(points_2d, points_3d, K, dist):
    """Run PnP and return (R, t, rmse)."""
    obj, img = [], []
    for name in points_2d:
        if name in points_3d:
            obj.append(points_3d[name])
            img.append(points_2d[name])

    if len(obj) < 4:
        return None, None, 999

    obj_arr = np.array(obj, dtype=np.float64)
    img_arr = np.array(img, dtype=np.float64)

    ok, rv, tv = cv2.solvePnP(obj_arr, img_arr, K, dist,
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        # Try EPNP with RANSAC
        ok, rv, tv, _ = cv2.solvePnPRansac(
            obj_arr, img_arr, K, dist,
            flags=cv2.SOLVEPNP_EPNP, iterationsCount=500,
            reprojectionError=5.0, confidence=0.99)
    if not ok:
        return None, None, 999

    R, _ = cv2.Rodrigues(rv)
    proj, _ = cv2.projectPoints(obj_arr, rv, tv, K, dist)
    errs = np.linalg.norm(proj.reshape(-1, 2) - img_arr, axis=1)
    rmse = float(np.sqrt(np.mean(errs**2)))

    return R, tv.ravel(), rmse


def compose(G_R_C, G_t_C, C_R_B, C_t_B):
    """Compose G_T_B = inv(C_T_G) * C_T_B."""
    # C_T_G → G_T_C
    C_R_G = np.linalg.inv(G_R_C)
    C_t_G = -C_R_G @ G_t_C
    G_R_C_mat = np.linalg.inv(C_R_G)
    G_t_C_vec = -G_R_C_mat @ C_t_G

    # G_R_B = G_R_C * C_R_B
    G_R_B = G_R_C_mat @ C_R_B
    return G_R_B


# ====================================================================
# Test A: Synthetic data
# ====================================================================

def test_A():
    """Set known yaw/pitch/roll, project 2D, recover via PnP+compose."""
    print("=" * 60)
    print("Test A: Synthetic data recovery")
    print("=" * 60)

    K, dist, b_pts_3d, g_pts_3d = load_configs()

    # Known aircraft pose in board frame
    yaw_true, pitch_true, roll_true = 10.0, 5.0, -3.0
    G_R_B_true = euler_to_R(yaw_true, pitch_true, roll_true)
    G_t_B_true = np.array([100.0, 50.0, -20.0])  # arbitrary translation

    # Known board pose (board in front of camera)
    C_R_G = np.eye(3)
    C_t_G = np.array([0.0, 0.0, 500.0])  # 500mm away

    # Compute C_T_B = C_T_G * G_T_B
    C_R_B = C_R_G @ G_R_B_true
    C_t_B = C_R_G @ G_t_B_true + C_t_G

    # Project B-frame points to camera image
    b_2d = project_points(b_pts_3d, K, dist, C_R_B, C_t_B)

    # Project G-frame points (board) to camera image
    # Simulate: board PnP gives C_T_G
    g_2d = project_points(g_pts_3d, K, dist, C_R_G, C_t_G)

    # --- Recover board pose ---
    # Board PnP: image ← G-frame
    G_R_C_est, G_t_C_est, g_rmse = pnp(g_2d, g_pts_3d, K, dist)
    print(f"\n  Board PnP RMSE: {g_rmse:.4f} px")

    # --- Recover aircraft pose ---
    # Aircraft PnP: image ← B-frame
    C_R_B_est, C_t_B_est, b_rmse = pnp(b_2d, b_pts_3d, K, dist)
    print(f"  Aircraft PnP RMSE: {b_rmse:.4f} px")

    if G_R_C_est is None or C_R_B_est is None:
        print("  FAIL: PnP failed")
        return False

    # --- Compose ---
    G_R_B_est = compose(G_R_C_est, G_t_C_est, C_R_B_est, C_t_B_est)
    roll_est, pitch_est, yaw_est = R_to_euler(G_R_B_est)

    yaw_err = abs(yaw_est - yaw_true)
    pitch_err = abs(pitch_est - pitch_true)
    roll_err = abs(roll_est - roll_true)

    print(f"\n  True:     yaw={yaw_true:.4f}  pitch={pitch_true:.4f}  roll={roll_true:.4f}")
    print(f"  Recovered: yaw={yaw_est:.4f}  pitch={pitch_est:.4f}  roll={roll_est:.4f}")
    print(f"  Error:     yaw={yaw_err:.4f}  pitch={pitch_err:.4f}  roll={roll_err:.4f}")

    passed = max(yaw_err, pitch_err, roll_err) < 0.01
    print(f"\n  {'PASSED' if passed else 'FAILED'} (threshold < 0.01 deg)")
    return passed


# ====================================================================
# Test B: Zero pose
# ====================================================================

def test_B():
    """Align aircraft B-frame with board G-frame. Expect near-zero angles."""
    print("\n" + "=" * 60)
    print("Test B: Zero pose test")
    print("=" * 60)

    K, dist, b_pts_3d, g_pts_3d = load_configs()

    # B-frame aligned with G-frame: G_R_B = I
    G_R_B_true = np.eye(3)
    G_t_B_true = np.array([0.0, 0.0, 0.0])

    # Board in front of camera
    C_R_G = np.eye(3)
    C_t_G = np.array([0.0, 0.0, 500.0])

    # C_T_B = C_T_G * G_T_B = C_T_G (since G_T_B = I)
    C_R_B = C_R_G @ G_R_B_true
    C_t_B = C_R_G @ G_t_B_true + C_t_G

    b_2d = project_points(b_pts_3d, K, dist, C_R_B, C_t_B)
    g_2d = project_points(g_pts_3d, K, dist, C_R_G, C_t_G)

    G_R_C_est, G_t_C_est, g_rmse = pnp(g_2d, g_pts_3d, K, dist)
    C_R_B_est, C_t_B_est, b_rmse = pnp(b_2d, b_pts_3d, K, dist)

    print(f"\n  Board PnP RMSE: {g_rmse:.4f} px")
    print(f"  Aircraft PnP RMSE: {b_rmse:.4f} px")

    if G_R_C_est is None or C_R_B_est is None:
        print("  FAIL: PnP failed")
        return False

    G_R_B_est = compose(G_R_C_est, G_t_C_est, C_R_B_est, C_t_B_est)
    roll, pitch, yaw = R_to_euler(G_R_B_est)

    print(f"\n  Recovered: yaw={yaw:.4f}  pitch={pitch:.4f}  roll={roll:.4f}")
    print(f"  Expected:  0.0  0.0  0.0")

    passed = max(abs(yaw), abs(pitch), abs(roll)) < 0.01
    print(f"\n  {'PASSED' if passed else 'FAILED'} (threshold < 0.01 deg)")
    return passed


# ====================================================================
# Test C: Single-axis sign
# ====================================================================

def test_C():
    """Test each axis independently: +5 deg should only affect target axis."""
    print("\n" + "=" * 60)
    print("Test C: Single-axis sign test")
    print("=" * 60)

    K, dist, b_pts_3d, g_pts_3d = load_configs()

    tests = [
        ('yaw +5',   5.0,  0.0,  0.0, 0),  # yaw changes, pitch/roll stay ~0
        ('pitch +5', 0.0,  5.0,  0.0, 1),  # pitch changes, yaw/roll stay ~0
        ('roll +5',  0.0,  0.0,  5.0, 2),  # roll changes, yaw/pitch stay ~0
    ]

    all_passed = True
    for label, y, p, r, target_axis in tests:
        G_R_B_true = euler_to_R(y, p, r)
        G_t_B_true = np.array([0.0, 0.0, 0.0])

        C_R_G = np.eye(3)
        C_t_G = np.array([0.0, 0.0, 500.0])

        C_R_B = C_R_G @ G_R_B_true
        C_t_B = C_R_G @ G_t_B_true + C_t_G

        b_2d = project_points(b_pts_3d, K, dist, C_R_B, C_t_B)
        g_2d = project_points(g_pts_3d, K, dist, C_R_G, C_t_G)

        G_R_C_est, G_t_C_est, _ = pnp(g_2d, g_pts_3d, K, dist)
        C_R_B_est, C_t_B_est, _ = pnp(b_2d, b_pts_3d, K, dist)

        if G_R_C_est is None or C_R_B_est is None:
            print(f"  {label}: FAIL (PnP)")
            all_passed = False
            continue

        G_R_B_est = compose(G_R_C_est, G_t_C_est, C_R_B_est, C_t_B_est)
        roll_e, pitch_e, yaw_e = R_to_euler(G_R_B_est)
        angles = [yaw_e, pitch_e, roll_e]  # [0]=yaw, [1]=pitch, [2]=roll

        errors = [abs(angles[i] - [y, p, r][i]) for i in range(3)]
        target_err = errors[target_axis]
        cross_err = sum(errors) - target_err

        print(f"  {label}: yaw={yaw_e:+.4f} pitch={pitch_e:+.4f} roll={roll_e:+.4f}  "
              f"target_err={target_err:.4f} cross_err={cross_err:.4f}")

        # Target axis should be within 0.01 of expected
        # Cross axes should be within 0.01 of zero
        passed = target_err < 0.01 and errors[(target_axis+1)%3] < 0.01 and errors[(target_axis+2)%3] < 0.01
        if not passed:
            print(f"    FAILED")
            all_passed = False

    print(f"\n  {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


# ====================================================================
# Test 1.4: Validation — reject G-frame points
# ====================================================================

def test_validation():
    """Verify that estimate_aircraft_pose.py rejects G-frame points."""
    print("\n" + "=" * 60)
    print("Test 1.4: G-frame rejection")
    print("=" * 60)

    import subprocess

    # Try with G-frame points (should fail with ValueError)
    py = sys.executable  # use same Python as test
    r = subprocess.run(
        [py, 'tools/estimate_aircraft_pose.py',
         '--config', 'configs/cameras/camera_20mm_far.yaml',
         '--aircraft-2d',
         'annotations/aircraft_2d/Pic_2026_07_09_193004_131_points.yaml',
         '--aircraft-3d', 'configs/aircraft_points_G_reference.yaml',
         '-o', 'output/test_g_reject.csv'],
        capture_output=True, text=True)
    g_rejected = ('aircraft body frame (B)' in r.stdout
                  or 'aircraft body frame (B)' in r.stderr)

    # Try with B-frame points (should succeed with inliers)
    r2 = subprocess.run(
        [py, 'tools/estimate_aircraft_pose.py',
         '--config', 'configs/cameras/camera_20mm_far.yaml',
         '--aircraft-2d',
         'annotations/aircraft_2d/Pic_2026_07_09_193004_131_points.yaml',
         '--aircraft-3d', 'configs/aircraft_points_B.yaml',
         '-o', 'output/test_b_accept.csv'],
        capture_output=True, text=True)
    b_accepted = 'inliers' in r2.stdout

    print(f"  G-frame points rejected: {'PASS' if g_rejected else 'FAIL'}")
    print(f"  B-frame points accepted: {'PASS' if b_accepted else 'FAIL'}")

    # Cleanup
    for f in ['output/test_g_reject.csv', 'output/test_b_accept.csv']:
        Path(f).unlink(missing_ok=True)

    return g_rejected and b_accepted


# ====================================================================
# Main
# ====================================================================

if __name__ == '__main__':
    results = {
        'Test A (synth recovery)': test_A(),
        'Test B (zero pose)': test_B(),
        'Test C (single axis)': test_C(),
        'Test 1.4 (validation)': test_validation(),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_ok = True
    for name, ok in results.items():
        print(f"  {name}: {'PASSED' if ok else 'FAILED'}")
        if not ok: all_ok = False

    print(f"\n  Overall: {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    sys.exit(0 if all_ok else 1)
