"""Detect calibration board circles → output 2D coordinates."""
import sys, cv2, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sls_calib.config_validator import load_camera_config, validate_image_size
from sls_calib.board_detector import detect_and_assign_board

def main():
    import argparse
    p = argparse.ArgumentParser(
        description='Detect calibration board pattern points')
    p.add_argument('image', help='Calibration board image')
    p.add_argument('--config', required=True,
                   help='Camera config (e.g. configs/cameras/camera_20mm_far.yaml)')
    p.add_argument('--output', '-o', default=None,
                   help='Output CSV (auto-generated if not set)')
    args = p.parse_args()

    cfg, K, dist = load_camera_config(args.config)

    img = cv2.imread(args.image)
    if img is None:
        print(f'Cannot read: {args.image}'); sys.exit(1)

    validate_image_size(img, args.config)

    result = detect_and_assign_board(img, K, dist)

    if not result.success:
        print(f'Detection failed: {result}')
        sys.exit(1)

    out_path = args.output or f'output/{Path(args.image).stem}_board2d.csv'
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('point_id,u,v\n')
        for i, (px, py) in enumerate(result.image_points):
            f.write(f'B{i+1:03d},{px:.3f},{py:.3f}\n')

    print(f'Detection complete: {result.assigned_count}/{result.detected_count} '
          f'valid points, RMSE={result.assignment_rmse:.3f}px -> {out_path}')

if __name__ == '__main__':
    main()
