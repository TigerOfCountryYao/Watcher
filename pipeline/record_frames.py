from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from pipeline.screen_capture import ScreenCapture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record frame-by-frame screenshots for dataset building.")
    parser.add_argument("--mode", default="window", choices=["desktop", "monitor", "window"])
    parser.add_argument("--window-title", default="微信|WeChat")
    parser.add_argument("--monitor", type=int, default=1)
    parser.add_argument("--output-dir", default="data/recordings/wechat_session")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--prefix", default="frame")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = ScreenCapture(
        mode=args.mode,
        monitor=args.monitor,
        window_title=args.window_title,
    )

    interval = 1.0 / max(args.fps, 1e-6)
    manifest_path = output_dir / "frames_manifest.jsonl"

    print(f"recording to: {output_dir}")
    print(f"mode={args.mode} window_title={args.window_title} fps={args.fps}")
    print("stop with: Ctrl+C")

    frame_idx = 0
    start_time = time.time()
    try:
        with manifest_path.open("a", encoding="utf-8") as manifest_file:
            while True:
                frame = capture.get_frame()
                if frame_idx == 0:
                    print(f"matched_window={capture.last_window_title or '(n/a)'}")
                    print(f"capture_region={capture.last_region}")
                    print(f"frame_size={frame.shape[1]}x{frame.shape[0]}")
                timestamp = time.time()
                filename = f"{args.prefix}_{frame_idx:06d}.png"
                frame_path = output_dir / filename
                cv2.imwrite(str(frame_path), frame)

                record = {
                    "frame_index": frame_idx,
                    "timestamp": timestamp,
                    "relative_path": filename,
                    "window_title": args.window_title,
                    "mode": args.mode,
                }
                manifest_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest_file.flush()

                frame_idx += 1
                elapsed = time.time() - timestamp
                time.sleep(max(0.0, interval - elapsed))
    except KeyboardInterrupt:
        duration = time.time() - start_time
        print(f"\nstopped. frames={frame_idx} duration={duration:.2f}s")
        print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
