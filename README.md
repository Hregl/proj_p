# SLS 标定工具包

使用编码（ArUco）标记和圆点标靶进行相机标定、运动恢复结构（SfM）和飞行器姿态估计。

## 项目结构

```
sls_calib/           # Python 包（核心算法）
  marker_detector.py   圆形标记检测 + 亚像素精化
  camera_calib.py      单相机内参标定（SLS 圆点网格）
  coded_marker.py      ArUco 编码标记: 检测、PnP、生成
  sfm_pipeline.py      多视图 SfM: 重建 + 捆绑调整
  stereo_calib.py      双目标定 + 立体校正
  pipeline.py          端到端流水线运行器

tools/               # 命令行入口
  generate_markers.py  打印 ArUco 标记页 / ChArUco 标定板
  run_calibration.py   单相机内参标定
  run_sfm.py           多视图 SfM 重建
  run_stereo_calib.py  双目系统标定
  run_pipeline.py      完整流水线（标定 → SfM → 姿态）

tests/               # 精度 / 回归测试
data/                # 图像和生成的标记（不跟踪）
output/              # 流水线结果（不跟踪）
```

## 安装

```bash
git clone https://github.com/Hregl/proj_p.git
cd proj_p
python -m venv venv
venv\Scripts\activate         # Windows
# source venv/bin/activate    # Linux/macOS
pip install -r requirements.txt
```

依赖: Python 3.10+, OpenCV 4.10+, NumPy 2.0+, SciPy 1.14+。

---

## 快速验证

```bash
# 在示例标定图像上测试标记检测
python tests/test_marker_detector.py data/p1.png
# 预期: 检测到 38 个标记

# 测试 ArUco 检测（生成标记页并检测）
python tests/test_coded_marker.py
# 预期: 8/8 标记检测成功, 通过

# 在合成数据上测试 SfM 精度
python tests/test_sfm_pipeline.py
# 预期: 平均 3D 误差 < 1 mm, 通过

# 在合成数据上测试双目标定
python tests/test_stereo_calib.py
# 预期: 旋转误差 < 0.5°, 通过
```

---

## 端到端工作流程（真实照片）

以下是飞行器姿态估计任务的完整流水线:

```
相机 1                           相机 2
(多视角)                         (单次任意角度)
     │                              │
     ├─ SfM 重建 ──────┐            │
     │  (阶段 A)       │            │
     │                 ▼            │
     │           3D 标记坐标        │
     │           (地面真值)         │
     │                 │            │
     │                 └────────────┤
     │                              │
     ▼                              ▼
   飞行器姿态 ←───── PnP ──── 单张图像
   相对于地面                    (阶段 B)
```

### 步骤 0: 准备标记

1. **生成用于打印的 ArUco 标记:**

   ```bash
   python tools/generate_markers.py --dict 4x4_50 --ids 0 1 2 3 --size 300 -o markers_ground.png
   python tools/generate_markers.py --dict 4x4_50 --ids 4 5 6 7 8 --size 200 -o markers_aircraft.png
   ```

2. **打印** 标记到白纸上（哑光纸，不要光面 — 光面纸反光太强，会导致检测失败）。

3. **粘贴** 地面标记平放在地板上（定义坐标基准）。
   将飞行器标记粘贴到模型的已知位置上。

4. **（可选）生成 ChArUco 标定板:**

   ```bash
   python tools/generate_markers.py --charuco --squares 5 7 -o charuco_board.png
   ```

### 步骤 1: 标定相机内参

在运行 SfM 或 PnP 之前，需要获取每个相机的内参（焦距、主点、畸变）。

**方案 A — 使用 SLS 圆点标靶**（如 `data/p1.png`）:

从 10-15 个不同角度拍摄标定标靶（覆盖整个视场，包含倾斜角度）。然后:

```bash
python tools/run_calibration.py data/calibration/*.png --circle-interval 35 -o camera.npz
```

**方案 B — 使用棋盘格:**

使用 `run_stereo_calib.py` 工具（内置棋盘格检测功能）分别标定每个相机，或直接使用 OpenCV 的 `cv2.calibrateCamera`。

**方案 C — 使用 ChArUco 板**（标记部分遮挡时最鲁棒）:

与棋盘格相同 — `run_stereo_calib.py` 工具支持 `--pattern charuco`。

### 步骤 2: 多视图 SfM（阶段 A — 地面真值）

使用 **相机 1** 从设置场景周围的 **6-15 个不同角度**拍摄场景（地面 + 飞行器）。确保:

- 每张照片至少能看到 **4-6 个 ArUco 标记**（包括地面和飞行器标记）
- 连续视图之间保持 **约 60-70% 的重叠**
- **变换高度和角度**（不要全部从同一仰角拍摄）
- **良好的光照** — 漫射、均匀的照明；避免标记上的直接反射
- **对焦清晰** — 运动模糊会严重影响亚像素精度

将照片放在 `data/sfm/` 中，然后:

```bash
python tools/run_sfm.py data/sfm/view_*.png \
    --camera camera.npz \
    --ground-ids 0 1 2 3 \
    --marker-size 0.03 \
    -o sfm_result.npz
```

**工作流程:**
1. 在每张图像中检测 ArUco 标记（ID 提供自动对应关系）
2. 选择最佳图像对进行初始化
3. 本质矩阵 → 相对姿态 → 三角测量 → 初始 3D 点
4. 通过 PnP 增量注册其余视图
5. 捆绑调整同时优化所有相机姿态和 3D 点
6. 地面标记定义 XY 平面（Z=0）；世界原点为其质心

**输出:** `sfm_result.npz` 包含:
- `marker_ids`: ArUco ID 数组
- `points_3d`: (N, 3) 数组，3D 标记位置，单位为 **米**（世界坐标系）
- `reproj_error`: 最终 RMS 重投影误差，单位为像素

误差 < 1.0 px 为优秀；< 2.0 px 可接受。误差较高通常意味着标定质量差、运动模糊或角度覆盖不足。

### 步骤 3: 飞行器姿态估计（阶段 B — 推理）

使用 **相机 2**（可以是不同的相机，同样需要标定）从任意角度拍摄一张**单张照片**，同时拍到地面和飞行器标记。

```python
import cv2
import numpy as np
from sls_calib import CodedMarkerDetector, MultiViewSfM

# 加载 SfM 结果
sfm_data = np.load("sfm_result.npz", allow_pickle=True)
marker_ids = sfm_data["marker_ids"]
points_3d = {int(mid): tuple(p) for mid, p in zip(marker_ids, sfm_data["points_3d"])}

# 加载相机 2 的标定参数
cam2 = np.load("camera2.npz")
K2 = cam2["camera_matrix"]
dist2 = cam2["dist_coeffs"]

# 在推理图像中检测标记
img = cv2.imread("data/inference/cam2_shot.png")
detector = CodedMarkerDetector("4x4_50")
markers, _ = detector.detect(img)

# PnP: 从地面标记求解相机姿态
# 使用 SfM 得到的已知 3D 位置
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
    print(f"相机 2 在世界坐标系中的位置: {(-R_cam.T @ tvec).ravel()} m")
```

### 步骤 4（可选）: 双目标定

如果需要两个相机之间的几何关系（用于立体深度估计或交叉验证），将它们标定为双目系统:

```bash
python tools/run_stereo_calib.py \
    --left data/stereo/left_*.png \
    --right data/stereo/right_*.png \
    --pattern chessboard --pattern-size 9 6 --square-size 0.025 \
    -o stereo_params.npz
```

这将计算两个相机之间的相对旋转 R 和平移 T，以及用于极线对齐的校正映射。`--pattern` 参数支持 `chessboard`、`circles`（SLS 圆点网格）和 `charuco`。

### 一键运行（配置文件）

创建 `config.json`:

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

然后运行:

```bash
python tools/run_pipeline.py config.json
```

或使用预先标定的内参:

```bash
python tools/run_pipeline.py --calib camera.npz \
    --sfm-images data/sfm/*.png \
    --ground-ids 0 1 2 3 --aircraft-ids 4 5 6 7 8 \
    --marker-size 0.03
```

---

## 标记布置指南

### 地面标记（推荐 4-6 个）

- 放置在**平坦、水平**的表面（地板或桌面）上
- 间距 **30-80 cm**（间距越大，角度分辨率越好）
- **至少 1 个标记应与其他标记偏移**（不能全部共线），
  防止地面平面法线产生歧义
- 使用较大的标记（5-10 cm）以便在远距离也能检测到
- 按惯例 ArUco ID 0-3 保留给地面标记

### 飞行器标记（推荐 5-10 个）

- 分布在飞行器表面上，**避免对称布置**
- 至少 3 个标记应**非共线**（用于刚体姿态求解）
- 使用较小的标记（2-5 cm）以适应模型尺寸
- 不同表面上的标记（机翼顶面、机身侧面、尾翼）提供最佳的
  姿态约束

### 拍照技巧

| 因素 | 建议 |
|---|---|
| 视图数量（SfM） | 8-15 |
| 角度覆盖 | 围绕场景至少 120° |
| 重叠度 | 连续视图之间 60-80% |
| 光照 | 漫射、均匀；避免阳光直射或高光反射 |
| 对焦 | 清晰；运动模糊是精度的头号杀手 |
| 分辨率 | 最小标记至少 20 px 宽 |
| 相机设置 | 固定对焦、固定光圈、固定白平衡 |

---

## 包 API 参考

```python
from sls_calib import (
    # 标记检测
    SLSMarkerDetector,           # 圆形圆点（轮廓 + 亚像素）
    CodedMarkerDetector,         # 带 ID 的 ArUco 编码标记
    UnifiedMarkerTracker,        # 组合两个检测器

    # 标定
    CalibImage, Calibrator,      # SLS 圆点网格标定

    # SfM
    MultiViewSfM, View,          # 多视图重建 + BA

    # 双目
    StereoCalibrator,            # 双目标定 + 校正
    calibrate_stereo_rig,        # 一键便捷函数

    # 流水线
    CalibrationPipeline,         # 端到端编排

    # 标记生成
    generate_marker_image,       # 单个 ArUco 标记
    generate_marker_sheet,       # 可打印的标记网格
    generate_charuco_board,      # ChArUco 标定板
)
```

## 故障排除

| 症状 | 可能原因 | 解决方法 |
|---|---|---|
| 检测到 0 个 ArUco 标记 | 光照差 / 模糊 / 标记太小 | 增大标记尺寸，改善光照，检查对焦 |
| SfM 初始化失败（"no suitable pair"） | 视图之间共享标记不够 | 增加重叠度，使用更多标记 |
| 重投影误差高（> 3 px） | 内参差或检测噪声大 | 重新标定相机，检查运动模糊 |
| 地面平面法线错误 | 地面标记不共面或识别错误 | 核实地面标记 ID，检查平整度 |
| 飞行器姿态翻转/跳动 | 标记太对称 | 采用非对称标记布置 |
| `ImportError` on `sls_calib` | 从错误的目录运行 | 始终从 `d:/proj_p/`（项目根目录）运行 |

## 参考文献

- Hartley & Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed.
- OpenCV 文档: [Camera Calibration](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- Garrido-Jurado et al., "Automatic generation and detection of highly reliable
  fiducial markers under occlusion", *Pattern Recognition*, 2014 (ArUco)
