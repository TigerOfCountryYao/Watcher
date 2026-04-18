from __future__ import annotations

import argparse
import ctypes
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from pipeline.screen_capture import ScreenCapture

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
VK_F8 = 0x77


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record frame-by-frame screenshots for dataset building.")
    parser.add_argument("--mode", default="window", choices=["desktop", "monitor", "window"])
    parser.add_argument("--window-title", default="微信")
    parser.add_argument("--monitor", type=int, default=1)
    parser.add_argument("--output-root", default="data/recordings")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--start-delay", type=float, default=3.0)
    parser.add_argument("--prefix", default="frame")
    return parser.parse_args()




def make_session_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).resolve()

    output_root = Path(args.output_root).resolve()
    title = args.window_title if args.mode == "window" and args.window_title else f"monitor{args.monitor}"
    safe_title = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff_-]+", "_", title).strip("_") or "session"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = f"{args.mode}_{safe_title}_{timestamp}"
    return output_root / session_name


def click_region_center(region: dict[str, int] | None) -> None:
    if not region:
        return

    center_x = int(region["left"] + region["width"] / 2)
    center_y = int(region["top"] + region["height"] / 2)
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.SetCursorPos(center_x, center_y)
    user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def is_stop_hotkey_pressed() -> bool:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    return bool(user32.GetAsyncKeyState(VK_F8) & 0x0001)


def save_image(frame_path: Path, frame: np.ndarray) -> None:
    suffix = frame_path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(suffix, frame)
    if not ok:
        raise RuntimeError(f"failed to encode image: {frame_path}")
    encoded.tofile(str(frame_path))


def read_image(frame_path: Path) -> np.ndarray | None:
    if not frame_path.exists():
        return None
    data = np.fromfile(str(frame_path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def filter_static_frames(output_dir: Path, manifest_path: Path) -> tuple[int, int]:
    records: list[dict[str, object]] = []
    if not manifest_path.exists():
        return 0, 0

    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        for line in manifest_file:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    kept_records: list[dict[str, object]] = []
    removed = 0
    prev_kept_frame: np.ndarray | None = None
    readable_frames = 0

    for record in records:
        relative_path = str(record["relative_path"])
        frame_path = output_dir / relative_path
        if not frame_path.exists():
            continue

        frame = read_image(frame_path)
        if frame is None:
            continue
        readable_frames += 1

        if prev_kept_frame is not None and np.array_equal(frame, prev_kept_frame):
            frame_path.unlink(missing_ok=True)
            removed += 1
            continue

        kept_records.append(record)
        prev_kept_frame = frame

    if readable_frames == 0:
        return 0, 0

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for new_index, record in enumerate(kept_records):
            record["frame_index"] = new_index
            manifest_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return len(kept_records), removed


def main() -> None:
    args = parse_args()
    output_dir = make_session_output_dir(args)
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
    print(f"start_delay={args.start_delay}s")
    print("stop with: Ctrl+C or F8")
    capture.prepare()
    if capture.last_window_title:
        print(f"prepared_window={capture.last_window_title}")
        print(f"prepared_region={capture.last_region}")
    time.sleep(max(args.start_delay, 0.0))
    click_region_center(capture.last_region)
    time.sleep(0.2)

    frame_idx = 0
    start_time = time.time()
    try:
        with manifest_path.open("a", encoding="utf-8") as manifest_file:
            while True:
                if is_stop_hotkey_pressed():
                    raise KeyboardInterrupt
                frame = capture.get_frame()
                if frame_idx == 0:
                    print(f"matched_window={capture.last_window_title or '(n/a)'}")
                    print(f"capture_region={capture.last_region}")
                    print(f"frame_size={frame.shape[1]}x{frame.shape[0]}")
                timestamp = time.time()
                filename = f"{args.prefix}_{frame_idx:06d}.png"
                frame_path = output_dir / filename
                save_image(frame_path, frame)

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
        kept_frames, removed_frames = filter_static_frames(output_dir, manifest_path)
        print(f"\nstopped. frames={frame_idx} duration={duration:.2f}s")
        print(f"filtered_static_frames={removed_frames} kept_frames={kept_frames}")
        print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
