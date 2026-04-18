from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


DEFAULT_SHARDS = [
    "data/train-00005-of-00016.parquet",
    "data/train-00007-of-00016.parquet",
    "data/train-00012-of-00016.parquet",
]


def sanitize_name(name: str) -> str:
    keep = []
    for char in name:
        if char.isalnum() or char in ("-", "_"):
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def parse_events(raw_events: str) -> Any:
    try:
        return json.loads(raw_events)
    except json.JSONDecodeError:
        return raw_events


def write_embedded_screenshots(task_dir: Path, screenshots: list[dict[str, Any]]) -> list[str]:
    screenshots_dir = task_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    local_paths: list[str] = []
    for idx, shot in enumerate(screenshots):
        suffix = Path(shot.get("path") or f"{idx:03d}.png").suffix or ".png"
        local_path = screenshots_dir / f"{idx:03d}{suffix}"
        data = shot["bytes"]
        if isinstance(data, memoryview):
            data = data.tobytes()
        local_path.write_bytes(data)
        local_paths.append(str(local_path))
    return local_paths


def build_record(
    row: dict[str, Any],
    task_dir: Path,
    local_screenshots: list[str],
    output_root: Path,
) -> dict[str, Any]:
    return {
        "unique_data_id": row.get("unique_data_id"),
        "taskId": row.get("taskId"),
        "task_name": row.get("task_name"),
        "instruction": row.get("task_name"),
        "category": row.get("category"),
        "subCategory": row.get("subCategory"),
        "application_website": row.get("application_website"),
        "tags": row.get("tags"),
        "benchmark": row.get("benchmark"),
        "appType": row.get("appType"),
        "difficulty": row.get("difficulty"),
        "os": row.get("os"),
        "requires_login": row.get("requires_login"),
        "completedAt": row.get("completedAt"),
        "video_file": row.get("video_file"),
        "dom_snaps_file": row.get("dom_snaps_file"),
        "local_dir": str(task_dir.relative_to(output_root).as_posix()),
        "local_screenshots": [str(Path(path).relative_to(output_root).as_posix()) for path in local_screenshots],
        "local_events": str((task_dir / "events.json").relative_to(output_root).as_posix()),
    }


def download_shard(dataset: str, shard_path: str, output_root: Path) -> Path:
    return Path(
        hf_hub_download(
            repo_id=dataset,
            repo_type="dataset",
            filename=shard_path,
            local_dir=str(output_root / "hf_cache"),
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a small inspectable PSAI subset from parquet shards.")
    parser.add_argument("--dataset", default="anaisleila/computer-use-data-psai")
    parser.add_argument("--output-root", default="data/external/psai_subset")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--include-video", action="store_true")
    parser.add_argument("--include-dom", action="store_true")
    parser.add_argument("--shards", nargs="*", default=DEFAULT_SHARDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    count = 0

    for shard in args.shards:
        parquet_path = download_shard(args.dataset, shard, output_root)
        table = pq.read_table(parquet_path)
        for row in table.to_pylist():
            screenshots = row.get("screenshots") or []
            if len(screenshots) < 2:
                continue

            count += 1
            task_dir = output_root / f"{count:03d}_{sanitize_name(row['unique_data_id'])}"
            task_dir.mkdir(parents=True, exist_ok=True)

            local_screenshots = write_embedded_screenshots(task_dir, screenshots)
            events = parse_events(row.get("events", ""))
            (task_dir / "events.json").write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")

            record = build_record(row, task_dir, local_screenshots, output_root)
            records.append(record)
            print(f"[{count}] extracted task: {row['task_name']}")

            if count >= args.limit:
                manifest_path = output_root / "subset_manifest.json"
                manifest_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                print(f"saved manifest: {manifest_path}")
                return

    manifest_path = output_root / "subset_manifest.json"
    manifest_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
