"""
Canonical coordinate transforms and Euler conversions.

Single source of truth for the entire project.
compose_aircraft_pose.py and all tests MUST import from here.
"""
import numpy as np, math


# ====================================================================
# Rotation ↔ Euler (ZYX convention: yaw=Z, pitch=Y, roll=X)
# ====================================================================

def euler_to_R(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """ZYX Euler (degrees) → 3x3 rotation matrix."""
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


def R_to_euler(R: np.ndarray) -> tuple:
    """3x3 rotation matrix → (yaw_deg, pitch_deg, roll_deg) ZYX Euler."""
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        rx = math.atan2(R[2, 1], R[2, 2])
        ry = math.atan2(-R[2, 0], sy)
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        rx = math.atan2(-R[1, 2], R[1, 1])
        ry = math.atan2(-R[2, 0], sy)
        rz = 0.0
    return np.degrees(rz), np.degrees(ry), np.degrees(rx)  # yaw, pitch, roll


def rotation_angle_error(R_true: np.ndarray, R_est: np.ndarray) -> float:
    """Total rotation angle error (degrees) from R_true^T * R_est."""
    R_diff = R_true.T @ R_est
    trace_val = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(math.acos(trace_val)))


# ====================================================================
# Coordinate transforms (PnP inputs → G-frame outputs)
# ====================================================================

def invert_pose(C_R_W: np.ndarray, C_t_W: np.ndarray
                ) -> tuple:
    """Invert camera-to-world pose to world-to-camera.

    Given:  C_T_W  (world → camera:  X_cam = C_R_W * X_world + C_t_W)
    Return: W_T_C  (camera → world: X_world = W_R_C * X_cam + W_t_C)
    """
    W_R_C = np.linalg.inv(C_R_W)
    W_t_C = -W_R_C @ C_t_W
    return W_R_C, W_t_C


def compose_G_T_B(C_R_G: np.ndarray, C_t_G: np.ndarray,
                  C_R_B: np.ndarray, C_t_B: np.ndarray
                  ) -> tuple:
    """Compose aircraft pose in board frame.

    Given:
      C_T_G: board(G) → camera(C)   [from board PnP]
      C_T_B: aircraft body(B) → camera(C)  [from aircraft PnP]

    Returns:
      G_R_B: aircraft orientation in board frame
      G_t_B: aircraft position in board frame

    Formula:
      G_T_B = inv(C_T_G) * C_T_B
      G_R_B = G_R_C * C_R_B  where G_R_C = inv(C_R_G)
    """
    G_R_C, G_t_C = invert_pose(C_R_G, C_t_G)
    G_R_B = G_R_C @ C_R_B
    G_t_B = G_t_C + G_R_C @ C_t_B
    return G_R_B, G_t_B


def project_points(points_3d: dict, K: np.ndarray, dist: np.ndarray,
                   R: np.ndarray, t: np.ndarray) -> dict:
    """Project 3D points to 2D using camera model.

    Args:
        points_3d: {name: np.array([x, y, z])}
        K: 3x3 intrinsics
        dist: distortion coefficients
        R, t: camera pose (world→camera): X_cam = R * X_world + t

    Returns:
        {name: (u, v)} 2D projections
    """
    import cv2
    names = list(points_3d.keys())
    pts = np.array([points_3d[n] for n in names], dtype=np.float64)
    rvec, _ = cv2.Rodrigues(R)
    proj, _ = cv2.projectPoints(pts, rvec, t, K, dist)
    return {names[i]: (float(proj[i, 0, 0]), float(proj[i, 0, 1]))
            for i in range(len(names))}


def pnp(points_2d: dict, points_3d: dict, K: np.ndarray, dist: np.ndarray
        ) -> tuple:
    """Solve PnP: 2D image points + 3D object points → (C_R_W, C_t_W, rmse).

    Returns camera-to-world pose where X_cam = C_R_W * X_world + C_t_W.
    Returns (None, None, 999) on failure.
    """
    import cv2
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
        ok, rv, tv, _ = cv2.solvePnPRansac(
            obj_arr, img_arr, K, dist,
            flags=cv2.SOLVEPNP_EPNP, iterationsCount=500,
            reprojectionError=5.0, confidence=0.99)
    if not ok:
        return None, None, 999

    C_R_W, _ = cv2.Rodrigues(rv)
    # Reprojection RMSE
    proj, _ = cv2.projectPoints(obj_arr, rv, tv, K, dist)
    errs = np.linalg.norm(proj.reshape(-1, 2) - img_arr, axis=1)
    rmse = float(np.sqrt(np.mean(errs**2)))

    return C_R_W, tv.ravel(), rmse


def add_noise(points_2d: dict, sigma_px: float, seed: int = 0) -> dict:
    """Add Gaussian noise to 2D point coordinates."""
    rng = np.random.default_rng(seed)
    noisy = {}
    for name, (u, v) in points_2d.items():
        noisy[name] = (float(u + rng.normal(0, sigma_px)),
                        float(v + rng.normal(0, sigma_px)))
    return noisy
