# SLS 标定工具包 — 基于地面标志点坐标系的飞机姿态评估

使用圆形点标靶进行相机内参标定，基于**实测地面标志点**定义地面坐标系(G)，
通过多视图三角测量重建飞机标志点三维坐标，最终评估飞机相对于地面的姿态。

## 核心改进 (v3)

- **标定板仅用于相机内参标定**，不再定义世界坐标系
- **地面坐标系 G 由实测地面标志点定义**：明确原点、X/Y/Z 轴、单位和每个标志点 ID
- **四阶段流水线**：每阶段输出结果文件 + JSON 报告 + 可视化
- **统一坐标变换命名**：`A_T_B` 表示将 B 系坐标转换至 A 系

## 四阶段流水线

```
阶段 1：相机内参标定 → 阶段 2：地面 G 系标定 → 阶段 3：飞机标志点 3D 标定 → 阶段 4：姿态评估
```

| 阶段 | 输入 | 核心输出 | 关键质量指标 |
|---|---|---|---|
| 1. 内参标定 | 标定板图像 | `stage_01_intrinsics.npz` | 总 RMS、逐图 RMSE、有效图像数 |
| 2. 地面 G 系标定 | 内参、地面标志点 3D、2D 标注 | `stage_02_ground_poses.json` | PnP RMSE、内点数/比例、跨帧一致性 |
| 3. 飞机 3D 标定 | 多视图 2D 标注、阶段 2 位姿 | `aircraft_points_G.yaml`, `aircraft_points_B.yaml` | 观测数、三角化夹角、重投影误差、刚体距离残差 |
| 4. 姿态评估 | 内参、地面位姿、B 系点、2D 标注 | `G_T_B`、yaw/pitch/roll | PnP RMSE、重复性 std/p95、失败率 |

## 输出目录结构

```
output/<session>/
├── stage_01_intrinsics.npz
├── stage_01_calibration_report.json
├── stage_01_calibration_report.csv
├── stage_01_reprojection_residuals.png
├── stage_02/
│   ├── stage_02_ground_poses.json
│   └── stage_02_ground_pose_report.json
├── stage_03_aircraft_points_G.yaml
├── stage_03_aircraft_points_B.yaml
├── stage_03_B_frame_report.json
└── stage_04/
    ├── *_final_pose.csv
    └── stage_04_repeatability_report.json
```

## 项目结构

```
sls_calib/           # Python 包（核心算法）
  marker_detector.py   圆形标记检测 + 亚像素精化
  camera_calib.py      单相机内参标定（SLS 圆点网格）
  coded_marker.py      ArUco 编码标记: 检测、PnP、生成
  sfm_pipeline.py      多视图 SfM: 重建 + 捆绑调整
  stereo_calib.py      双目标定 + 立体校正
  ground_pose.py       地面坐标系位姿估计 (GroundPoseEstimator)
  transforms.py        坐标变换（欧拉角、姿态合成）
  board_detector.py    标定板圆点检测器
  config_validator.py  配置验证
  pipeline.py          端到端流水线运行器

tools/               # 命令行入口
  run_calibration.py        阶段 1: 相机内参标定
  estimate_ground_pose.py   阶段 2: 地面 G 系标定
  triangulate_aircraft_points.py  阶段 3a: 飞机点多视图三角测量
  convert_to_B_frame.py    阶段 3b: G 系→B 系转换
  estimate_aircraft_pose.py 阶段 4a: 飞机 PnP 姿态估计
  compose_aircraft_pose.py  阶段 4b: 合成最终姿态 G_T_B
  evaluate_repeatability.py 重复性评估
  visualize_axes.py         坐标轴可视化
  run_pipeline.py           四阶段流水线编排器

configs/
  calibration_board_points.yaml   标定板点坐标 (CALIB_BOARD 系)
  ground_markers_G.yaml           地面标志点 3D 坐标 (G 系) — 需实测
  experiment_ground_pipeline.yaml 实验配置文件
  aircraft_points_G.yaml         飞机点 G 系坐标 (阶段 3 输出)
  aircraft_points_B.yaml         飞机点 B 系坐标 (阶段 3 输出)
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

## 坐标约定

| 坐标系 | 符号 | 定义 | 来源 |
|---|---|---|---|
| 相机系 | C | 相机光心，Z 沿光轴 | 内参标定 |
| 地面系 | G | 由实测地面标志点定义，Z 向上 | 地面标志点测量 |
| 机体系 | B | 原点在机脊，X 尾→头，Y 左→右，Z 向上 | G 系点转换 |
| 标定板系 | CALIB_BOARD | 标定板局部坐标系（仅阶段 1 使用） | 标定板设计 |

**变换命名**: `A_T_B` 表示将 B 系坐标转换至 A 系的刚体变换。
- 例: `C_T_G` = 地面系→相机系 (PnP 直接输出)
- 例: `G_T_B` = 机体系→地面系 (最终姿态结果)

**欧拉角约定**: ZYX 顺序 (yaw-pitch-roll)，单位：度。

## 端到端实验流程

### 准备

1. **标定板**: 打印 SLS 圆形点标靶 (11×9 网格, 间距 25mm)
2. **地面标志点**: 在地面布设 ≥6 个标志点，用激光测距仪/全站仪实测 3D 坐标
3. **飞机标志点**: 在飞机模型上粘贴高对比度圆形标志点 (≥8 个，非对称分布)
4. **配置文件**:
   - 编辑 `configs/ground_markers_G.yaml` — 填入实测地面标志点坐标
   - 编辑 `configs/experiment_ground_pipeline.yaml` — 设置图像路径和阈值

### 一键运行

```bash
python tools/run_pipeline.py configs/experiment_ground_pipeline.yaml
```

或分阶段运行:

```bash
# 阶段 1: 相机内参标定
python tools/run_calibration.py data/calib/*.png --circle-interval 25 \
    -o output/experiment_01/

# 阶段 2: 地面 G 系标定
python tools/estimate_ground_pose.py \
    --config configs/cameras/camera_25mm.yaml \
    --ground-3d configs/ground_markers_G.yaml \
    --images data/ground_views/*.png \
    --annotations annotations/ground_2d/ \
    --output-dir output/experiment_01/stage_02/

# 阶段 3: 飞机标志点 3D 标定 (交互式 GUI)
python tools/triangulate_aircraft_points.py data/tri/*.png \
    --config configs/experiment_config.yaml \
    --ground-poses output/experiment_01/stage_02/stage_02_ground_poses.json \
    --point-names 机舱顶 左翼尖 右翼尖 机脊中部 \
                  左横尾翼尖 右横尾翼尖 左竖尾翼尖 右竖尾翼尖 \
    --output output/experiment_01/stage_03_aircraft_points_G.yaml

# 阶段 3b: B 系构建
python tools/convert_to_B_frame.py \
    --input output/experiment_01/stage_03_aircraft_points_G.yaml \
    --output output/experiment_01/stage_03_aircraft_points_B.yaml

# 阶段 4: 姿态评估 (逐帧)
python tools/estimate_aircraft_pose.py \
    --config configs/cameras/camera_25mm.yaml \
    --aircraft-3d configs/aircraft_points_B.yaml \
    --aircraft-2d annotations/aircraft_2d/MVIMG_20260707_202357_points.yaml

python tools/compose_aircraft_pose.py \
    --ground-pose output/stage_02/XXX_ground_pose.csv \
    --aircraft-pose output/XXX_aircraft_pose.csv \
    --output output/XXX_final_pose.csv

# 重复性评估
python tools/evaluate_repeatability.py output/*_final_pose.csv \
    --group-name experiment_01 \
    --output output/experiment_01/repeatability_report.json
```

## 质量门限

| 指标 | 优秀 | 可接受 | 差 |
|---|---|---|---|
| 标定 RMS | < 0.5 px | < 1.5 px | > 2.0 px |
| 地面 PnP RMSE | < 1.0 px | < 2.0 px | > 3.0 px |
| 三角化重投影误差 | < 1.0 px | < 2.0 px | > 3.0 px |
| 三角化夹角 | > 15° | > 8° | < 5° |
| 飞机 PnP RMSE | < 1.0 px | < 2.0 px | > 3.0 px |
| 重复性 (std) | < 5 arcmin | < 10 arcmin | > 30 arcmin |
| G→B→G 回环误差 | < 0.01 mm | < 0.1 mm | > 1.0 mm |
| det(G_R_B) | 1.000 ± 0.001 | 1.000 ± 0.01 | 偏离过大 |

## 误差解释边界

重投影误差低、重复性好，只能证明系统在当前数据上的**内部一致性和稳定性**；
它们不等价于绝对姿态准确度。

若需要声明例如 **1/60°（1 arcmin）绝对角度精度**，必须引入独立真值：
已知倾角治具、编码器、高精度电子水平仪，或经溯源的外部测量系统。

最终应报告：

```
θ_err = cos⁻¹((trace(R_gt^T · R_est) − 1) / 2)
```

并按 yaw/pitch/roll 分量和总旋转角分别统计均值、标准差、p95 与最大值。

## 数据采集建议

- **地面标志点**: 大范围分布，避免集中、共线；所有点虽在地面平面上，但二维布局必须覆盖拍摄区域
- **飞机标志点**: 固定使用**至少 8 个**非对称、空间分散的点
- **三角化视角**: 需要足够基线；夹角过小、观测次数过少和重投影残差过大的点应拒绝或补拍
- **元数据**: 所有照片应带 session、相机、焦距/镜头、分辨率等元数据

## 包 API 参考

```python
from sls_calib import (
    # 标记检测
    SLSMarkerDetector,           # 圆形圆点（轮廓 + 亚像素）
    CodedMarkerDetector,         # 带 ID 的 ArUco 编码标记

    # 标定
    CalibImage, Calibrator,      # SLS 圆点网格标定

    # 地面位姿
    GroundPoseEstimator,         # 地面标志点 PnP 位姿估计
    GroundPoseResult,            # 单帧结果
    load_ground_poses,           # 加载阶段 2 输出

    # 坐标变换
    compose_G_T_B,               # 合成 G_T_B
    R_to_euler,                  # 旋转矩阵 → 欧拉角
    euler_to_R,                  # 欧拉角 → 旋转矩阵
    rotation_angle_error,        # 旋转角误差

    # SfM
    MultiViewSfM, View,          # 多视图重建 + BA

    # 双目
    StereoCalibrator,            # 双目标定 + 校正

    # 流水线
    CalibrationPipeline,         # 端到端编排
)
```

## 故障排除

| 症状 | 可能原因 | 解决方法 |
|---|---|---|
| 标定 RMS > 2 px | 对焦差 / 运动模糊 / 检测失败 | 重新拍摄，检查光照 |
| 地面 PnP 失败 | 标志点匹配错误 / 3D 坐标不准 | 核实实测坐标，检查标注 ID |
| 三角化失败 | 有效视图 < 3 / 夹角太小 | 增加视角，扩大基线 |
| G→B→G 回环误差大 | G_R_B 不正交 | 检查参考点选择 |
| 重复性差 (>30 arcmin) | 标志点不足 / 标注噪声大 | 增加标志点，使用亚像素精化 |
| 欧拉角跳变 | 万向节死锁 (pitch ≈ ±90°) | 改用四元数或旋转矩阵直接比较 |

## 参考文献

- Hartley & Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed.
- OpenCV 文档: [Camera Calibration](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- Garrido-Jurado et al., "Automatic generation and detection of highly reliable
  fiducial markers under occlusion", *Pattern Recognition*, 2014 (ArUco)
