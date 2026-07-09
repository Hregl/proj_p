"""阶段4: 半自动标注飞机标志点2D坐标

使用方式:
  1. 在标定板+飞机同框的照片上点击飞机标志点
  2. 按数字键1-8选择点编号，鼠标左键点击标注
  3. 按 's' 保存, 按 'q' 退出
"""
import sys, yaml, cv2, numpy as np, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

class AircraftLabeler:
    def __init__(self, image_path, aircraft_yaml):
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise FileNotFoundError(f'无法读取: {image_path}')
        self.orig = self.img.copy()
        with open(aircraft_yaml, encoding='utf-8') as f:
            self.ac = yaml.safe_load(f)
        self.point_ids = list(self.ac['points'].keys())
        self.points = {}  # point_id -> (u,v)
        self.current_id = 0

    def run(self):
        cv2.namedWindow('标注飞机标志点')
        cv2.setMouseCallback('标注飞机标志点', self._click)
        print('=== 飞机标志点半自动标注 ===')
        print('  1-9: 选择点编号')
        print('  鼠标左键: 标注当前点的位置')
        print('  s: 保存到 annotations/aircraft_2d/')
        print('  q: 退出')
        print(f'  待标注点: {self.point_ids}')

        while True:
            display = self.img.copy()
            h, w = display.shape[:2]
            # 画已标注的点
            for pid, (u, v) in self.points.items():
                color = (0, 255, 0)
                cv2.circle(display, (int(u), int(v)), 5, color, -1)
                cv2.putText(display, pid, (int(u)+10, int(v)-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            # 当前选中的点提示
            if self.current_id < len(self.point_ids):
                pid = self.point_ids[self.current_id]
                region = self.ac['points'][pid].get('region','')
                cv2.putText(display, f'当前: {pid} ({region})', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('标注飞机标志点', display)
            key = cv2.waitKey(50) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                self._save()
            elif ord('1') <= key <= ord('9'):
                idx = key - ord('1')
                if idx < len(self.point_ids):
                    self.current_id = idx
                    print(f'选中: {self.point_ids[idx]}')

        cv2.destroyAllWindows()

    def _click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_id < len(self.point_ids):
                pid = self.point_ids[self.current_id]
                self.points[pid] = (float(x), float(y))
                cv2.circle(self.img, (x, y), 5, (0, 255, 0), -1)
                print(f'  标注: {pid} -> ({x}, {y})')
                self.current_id = min(self.current_id + 1, len(self.point_ids) - 1)

    def _save(self):
        out_dir = Path('annotations/aircraft_2d')
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / 'points.csv'
        with open(out_file, 'w') as f:
            f.write('point_id,u,v\n')
            for pid in self.point_ids:
                if pid in self.points:
                    u, v = self.points[pid]
                    f.write(f'{pid},{u:.3f},{v:.3f}\n')
        print(f'保存到: {out_file}')

def main():
    import argparse
    p = argparse.ArgumentParser(description='半自动标注飞机标志点')
    p.add_argument('image', help='包含飞机+标定板的照片')
    p.add_argument('--aircraft-yaml', default='configs/aircraft_points.yaml')
    args = p.parse_args()
    labeler = AircraftLabeler(args.image, args.aircraft_yaml)
    labeler.run()

if __name__ == '__main__': main()
