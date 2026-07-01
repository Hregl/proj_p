"""生成可打印的 ArUco 标记页和 ChArUco 标定板。

用法:
    python tools/generate_markers.py --dict 4x4_50 --ids 0 1 2 3 4 5 --size 300
    python tools/generate_markers.py --charuco --output charuco_board.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2
from sls_calib import generate_marker_sheet, generate_charuco_board, generate_marker_image


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate ArUco markers and ChArUco boards for printing")
    parser.add_argument("--dict", default="4x4_50",
                        help="ArUco dictionary (default: 4x4_50)")
    parser.add_argument("--ids", type=int, nargs="*",
                        help="Marker IDs (default: 0..15)")
    parser.add_argument("--size", type=int, default=300,
                        help="Marker pixel size (default: 300)")
    parser.add_argument("--columns", type=int, default=4,
                        help="Markers per row (default: 4)")
    parser.add_argument("--charuco", action="store_true",
                        help="Generate a ChArUco board instead of marker sheet")
    parser.add_argument("--squares", type=int, nargs=2, default=[5, 7],
                        help="ChArUco squares (cols rows) (default: 5 7)")
    parser.add_argument("--output", "-o", default="aruco_sheet.png",
                        help="Output filename (default: aruco_sheet.png)")
    args = parser.parse_args()

    if args.charuco:
        board = generate_charuco_board(args.dict, args.squares[0], args.squares[1])
        board_img = board.generateImage((1400, 1000))
        cv2.imwrite(args.output, board_img)
        print(f"ChArUco 标定板: {args.output} ({args.squares[0]}x{args.squares[1]})")
    else:
        ids = args.ids if args.ids else list(range(16))
        sheet = generate_marker_sheet(args.dict, pixel_size=args.size,
                                      ids=ids, columns=args.columns)
        cv2.imwrite(args.output, sheet)
        rows = (len(ids) + args.columns - 1) // args.columns
        print(f"标记页: {args.output} ({len(ids)} 个标记, "
              f"{args.columns}x{rows} 网格, 每个 {args.size}px)")


if __name__ == "__main__":
    main()
