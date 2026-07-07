"""阶段2: 检测标定板圆点 → 输出2D坐标"""
import sys, yaml, cv2, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sls_calib import CalibImage, Calibrator

def main():
    import argparse
    p = argparse.ArgumentParser(description='Detect calibration board pattern points')
    p.add_argument('image', help='Calibration board image')
    p.add_argument('--interval', type=float, default=35, help='Circle spacing (mm)')
    p.add_argument('--threshold', type=float, default=0.55, help='Large circle threshold')
    p.add_argument('--output', '-o', default='annotations/board_2d/points.csv')
    args = p.parse_args()

    img = cv2.imread(args.image)
    if img is None: print(f'Cannot read: {args.image}'); sys.exit(1)

    ci = CalibImage(name='board', image=img, selected=True)
    calib = Calibrator()
    calib.extract_circles([ci], only_selected=False, smooth=True, debug=False)
    ci.create_display_circles()
    err = ci.find_circle_indices(args.interval, debug=False, large_circle_threshold=args.threshold)
    if err: print(f'Error: {err}'); sys.exit(1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        f.write('point_id,u,v\n')
        for i, ((px,py),(wx,wy,wz),ok,_) in enumerate(ci.circle_array):
            if ok:
                f.write(f'B{i+1:03d},{px:.3f},{py:.3f}\n')

    valid = sum(1 for _,_,ok,_ in ci.circle_array if ok)
    print(f'Detection complete: {valid}/99 valid points -> {args.output}')

if __name__ == '__main__': main()
