"""
Blur detection v2 — composite sharpness scoring via Sobel + Laplacian.

Calibration targets (white circles on dark board) have inherently low edge
content, so Laplacian variance alone has poor discrimination. This tool
combines three metrics into a normalized composite score (0-100):

  - Tenengrad (Sobel gradient magnitude) — edge strength
  - Laplacian variance — fine detail
  - Local contrast (std dev) — overall texture

Usage:
  python tools/detect_blur.py data/board_20mm/              # rank all images
  python tools/detect_blur.py data/board_20mm/ --top 20     # show worst 20
  python tools/detect_blur.py data/board_20mm/ --move 20    # move worst 20 to _blur/
  python tools/detect_blur.py data/board_20mm/ --keep 50    # keep best 50, move rest
"""
import sys, cv2, numpy as np, os, shutil
from pathlib import Path


def composite_sharpness(img: np.ndarray) -> dict:
    """Compute three sharpness metrics. Returns dict with raw + normalized scores."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    # 1. Tenengrad: mean of Sobel gradient magnitude
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    tenengrad = float(np.mean(grad_mag))

    # 2. Laplacian variance
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(lap.var())

    # 3. Local contrast: mean of local std dev (8x8 blocks)
    h, w = gray.shape
    # Pad to multiple of 8
    h8 = (h // 8) * 8
    w8 = (w // 8) * 8
    patches = gray[:h8, :w8].reshape(h8//8, 8, w8//8, 8).transpose(0, 2, 1, 3).reshape(-1, 64)
    local_std = float(np.mean(np.std(patches, axis=1)))

    return {
        'tenengrad': tenengrad,
        'lap_var': lap_var,
        'local_std': local_std,
    }


def main():
    import argparse
    p = argparse.ArgumentParser(description='Detect blurry photos (v2 — composite scoring)')
    p.add_argument('path', help='Directory of images')
    p.add_argument('--top', type=int, default=0,
                   help='Show only the worst N images')
    p.add_argument('--move', type=int, default=0,
                   help='Move worst N images to _blur/ subfolder')
    p.add_argument('--keep', type=int, default=0,
                   help='Keep best N, move rest to _blur/ (overrides --move)')
    p.add_argument('--threshold', type=float, default=None,
                   help='Manual composite score threshold (0-100)')
    p.add_argument('--list', action='store_true',
                   help='Only list blurry filenames (for scripting, use with --move)')
    args = p.parse_args()

    # Collect images
    pth = Path(args.path)
    if not pth.is_dir():
        print(f'Error: {args.path} is not a directory'); sys.exit(1)

    seen = set()
    files = []
    for ext in ['*.jpg', '*.png', '*.JPG', '*.JPEG', '*.bmp', '*.BMP']:
        for f in sorted(pth.glob(ext)):
            key = f.name.lower()
            if key not in seen:
                seen.add(key)
                files.append(f)

    if not files:
        print('No images found'); sys.exit(1)

    print(f'Scanning {len(files)} images...')

    # Compute all three metrics
    results = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        metrics = composite_sharpness(img)
        results.append({'file': f, 'shape': img.shape[:2], **metrics})

    if not results:
        print('No readable images'); sys.exit(1)

    # Normalize each metric to [0, 1] across the dataset
    for key in ['tenengrad', 'lap_var', 'local_std']:
        vals = np.array([r[key] for r in results])
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            for r in results:
                r[f'{key}_norm'] = (r[key] - vmin) / (vmax - vmin)
        else:
            for r in results:
                r[f'{key}_norm'] = 0.5

    # Composite score: weighted average of normalized metrics
    # Tenengrad is most reliable for calibration targets
    for r in results:
        r['score'] = (0.50 * r['tenengrad_norm'] +
                      0.25 * r['lap_var_norm'] +
                      0.25 * r['local_std_norm']) * 100

    # Sort by score ascending (worst first)
    results.sort(key=lambda r: r['score'])

    n = len(results)
    scores = np.array([r['score'] for r in results])

    print(f'\n{"="*65}')
    print(f'  Images: {n}')
    print(f'  Composite score range: {scores.min():.1f} - {scores.max():.1f}')
    print(f'  Median: {np.median(scores):.1f}  |  Mean: {scores.mean():.1f}')
    print(f'  Distribution: 10%={np.percentile(scores,10):.1f}  '
          f'25%={np.percentile(scores,25):.1f}  '
          f'75%={np.percentile(scores,75):.1f}  '
          f'90%={np.percentile(scores,90):.1f}')
    print(f'{"="*65}')

    # Determine how many to show/move
    n_blur = 0
    if args.keep > 0:
        n_blur = max(0, n - args.keep)
    elif args.move > 0:
        n_blur = args.move
    elif args.threshold is not None:
        n_blur = int(np.sum(scores < args.threshold))

    # Show results
    show_n = args.top if args.top > 0 else n
    show_n = min(show_n, n)
    if args.top == 0 and n_blur == 0 and args.threshold is None:
        show_n = n  # show all

    if show_n < n:
        print(f'\n  Showing worst {show_n} of {n}:')
    print(f"\n  {'Rank':>4}  {'Filename':<42s} {'Score':>6} {'Tenengrad':>10} {'Laplacian':>10} {'Contrast':>9}")

    # Auto-detect natural gap
    score_diffs = np.diff(scores)
    if len(score_diffs) > 0:
        gap_idx = int(np.argmax(score_diffs))
        natural_cut = int(np.percentile(scores, max(10, gap_idx / n * 100)))
    else:
        natural_cut = int(np.percentile(scores, 15))

    for i, r in enumerate(results[:show_n]):
        rank = i + 1
        # Mark low scorers
        if r['score'] < 25:
            flag = ' *** VERY BLURRY'
        elif r['score'] < 45:
            flag = ' ** blurry'
        elif r['score'] < natural_cut:
            flag = ' * soft'
        else:
            flag = ''
        print(f"  {rank:>4}: {r['file'].name:<42s} {r['score']:>5.1f}  "
              f"{r['tenengrad']:>9.2f}  {r['lap_var']:>9.1f}  {r['local_std']:>8.2f}{flag}")

    # Recommendation
    if n_blur == 0 and args.threshold is None:
        pct_cut = 25
        cut_score = int(np.percentile(scores, pct_cut))
        cut_n = max(1, int(n * pct_cut / 100))
        print(f'\n  Suggested cut: bottom {pct_cut}% (score < {cut_score}) = {cut_n} images')
        print(f'  Run with --move {cut_n} to remove them')

    # Move
    n_to_move = n_blur
    if args.keep > 0:
        n_to_move = max(0, n - args.keep)
    if args.move > 0:
        n_to_move = args.move
    if args.threshold is not None:
        n_to_move = int(np.sum(scores < args.threshold))

    if n_to_move > 0:
        blur_dir = pth / '_blur'
        blur_dir.mkdir(parents=True, exist_ok=True)
        moved = 0
        for r in results[:n_to_move]:
            dst = blur_dir / r['file'].name
            shutil.move(str(r['file']), str(dst))
            moved += 1
        print(f'\n  Moved {moved} images -> {blur_dir}/')

    if args.list and n_to_move > 0:
        for r in results[:n_to_move]:
            print(r['file'].name)


if __name__ == '__main__':
    main()
