from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def load_manifest(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_pairs_from_screenshots(
    manifest: list[dict[str, Any]],
    output_root: Path,
    pair_stride: int,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []

    for record in manifest:
        screenshots = record.get("local_screenshots") or []
        if len(screenshots) < 2:
            continue

        for idx in range(0, len(screenshots) - pair_stride, pair_stride):
            prev_frame = screenshots[idx]
            curr_frame = screenshots[idx + pair_stride]
            sample_id = f"{record['unique_data_id']}-pair-{idx:03d}"
            samples.append(
                {
                    "sample_id": sample_id,
                    "prev_frame": str((output_root / prev_frame).resolve().relative_to(Path.cwd()).as_posix()),
                    "frame": str((output_root / curr_frame).resolve().relative_to(Path.cwd()).as_posix()),
                    "instruction": record["instruction"],
                    "metadata": {
                        "source_dataset": "anaisleila/computer-use-data-psai",
                        "unique_data_id": record["unique_data_id"],
                        "taskId": record["taskId"],
                        "category": record["category"],
                        "application_website": record["application_website"],
                        "pair_index": idx,
                        "capture_source": "screenshots",
                    },
                }
            )
    return samples


def split_samples(
    samples: list[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    items = samples[:]
    random.Random(seed).shuffle(items)

    n = len(items)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_items = items[:train_end]
    val_items = items[train_end:val_end]
    test_items = items[val_end:]
    return train_items, val_items, test_items


def write_json(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build frame-pair manifests from a downloaded PSAI subset.")
    parser.add_argument("--input-root", default="data/external/psai_subset")
    parser.add_argument("--manifest", default="subset_manifest.json")
    parser.add_argument("--output-root", default="data/processed/psai_subset")
    parser.add_argument("--pair-stride", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()

    manifest = load_manifest(input_root / args.manifest)
    samples = build_pairs_from_screenshots(manifest, input_root, pair_stride=args.pair_stride)
    train_items, val_items, test_items = split_samples(
        samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    write_json(output_root / "all_unlabeled.json", samples)
    write_json(output_root / "train_unlabeled.json", train_items)
    write_json(output_root / "val_unlabeled.json", val_items)
    write_json(output_root / "test_unlabeled.json", test_items)

    print(f"samples={len(samples)} train={len(train_items)} val={len(val_items)} test={len(test_items)}")
    print(f"saved: {output_root}")


if __name__ == "__main__":
    main()
