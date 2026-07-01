"""端到端标定流水线命令行工具。

用法:
    python tools/run_pipeline.py config.json
    python tools/run_pipeline.py --data-dir data/scene --output-dir output/scene \
        --calib camera.npz --sfm-images data/scene/*.png \
        --ground-ids 0 1 2 3 --aircraft-ids 4 5 6
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2
from sls_calib import CalibrationPipeline


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="SLS calibration + SfM + pose estimation pipeline")

    # 配置文件模式
    parser.add_argument("config", nargs="?", help="JSON config file path")

    # 直接命令行模式
    parser.add_argument("--data-dir", default="data",
                        help="Data directory (default: data)")
    parser.add_argument("--output-dir", default="output",
                        help="Output directory (default: output)")
    parser.add_argument("--calib", help="Camera intrinsics .npz (skip calibration)")
    parser.add_argument("--sfm-images", nargs="*",
                        help="SfM view images (optional, for --calib mode)")
    parser.add_argument("--ground-ids", type=int, nargs="+", default=[],
                        help="ArUco IDs of ground markers")
    parser.add_argument("--aircraft-ids", type=int, nargs="+", default=[],
                        help="ArUco IDs of aircraft markers")
    parser.add_argument("--marker-size", type=float,
                        help="ArUco marker physical size (metres)")
    parser.add_argument("--aruco-dict", default="4x4_50",
                        help="ArUco dictionary (default: 4x4_50)")

    args = parser.parse_args()

    # 构建配置
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = {
            "data_dir": args.data_dir,
            "output_dir": args.output_dir,
            "marker_size_m": args.marker_size,
            "aruco_dict": args.aruco_dict,
            "ground_marker_ids": args.ground_ids,
            "aircraft_marker_ids": args.aircraft_ids,
        }

    pipeline = CalibrationPipeline(config)

    # 确定运行模式
    if args.calib:
        # 跳过标定，运行 SfM + 姿态估计
        pipeline.load_calibration(args.calib)

        sfm_images = None
        if args.sfm_images:
            sfm_images = [cv2.imread(p) for p in args.sfm_images
                          if cv2.imread(p) is not None]

        results = pipeline.run_all(calib_images=None, sfm_images=sfm_images)
    elif args.config:
        # 基于配置文件: 尝试从 data_dir 自动加载
        results = pipeline.run_all()
    else:
        parser.print_help()
        sys.exit(1)

    pipeline.export_results()
    print(f"\n结果已导出到 {pipeline.output_dir}/")


if __name__ == "__main__":
    main()
