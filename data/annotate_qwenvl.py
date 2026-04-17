from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SYSTEM_PROMPT = """You are labeling task-conditioned visual events for a screen agent.
Given a previous frame, a current frame, and one task instruction, decide whether the current frame contains a task-relevant event.

Instruction definition:
- The instruction is one concrete visual condition that should be checked on screen.
- It should refer only to what is visibly present or changes on screen.

Rules:
- Output label 1 only if the current frame shows a meaningful task-relevant change or reaches a target state for the instruction.
- Output label 0 if there is no relevant event for the instruction.
- Ignore irrelevant visual noise, cursor movement, compression artifacts, and background changes unless they matter for the task.
- Be conservative. If uncertain, prefer 0.
- Return strict JSON only.

JSON schema:
{"label": 0 or 1, "reason": "short explanation"}
"""


@dataclass
class Sample:
    prev_frame: str
    frame: str
    instruction: str
    sample_id: str | None = None
    metadata: dict[str, Any] | None = None


class QwenVLAnnotator:
    def __init__(
        self,
        model_name_or_path: str,
        cache_dir: str | None = None,
        device: str = "cuda",
        max_new_tokens: int = 128,
    ) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = device
        self.max_new_tokens = max_new_tokens

        torch_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            dtype=torch_dtype,
            device_map="auto" if device.startswith("cuda") else None,
            cache_dir=cache_dir,
        )
        if device == "cpu":
            self.model.to("cpu")
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
        )

    def annotate(self, sample: Sample) -> dict[str, Any]:
        prev_uri = Path(sample.prev_frame).resolve().as_uri()
        curr_uri = Path(sample.frame).resolve().as_uri()
        user_prompt = (
            f"Instruction: {sample.instruction}\n"
            "Determine whether the current frame contains a task-relevant event compared with the previous frame.\n"
            "Return strict JSON only."
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": prev_uri},
                    {"type": "image", "image": curr_uri},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        if self.device != "cpu":
            inputs = inputs.to(self.model.device)

        input_len = len(inputs.input_ids[0])
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        output_text = self.processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        parsed = parse_json_object(output_text)
        label = int(parsed["label"])
        if label not in (0, 1):
            raise ValueError(f"invalid label in model output: {parsed}")

        result = {
            "prev_frame": sample.prev_frame,
            "frame": sample.frame,
            "instruction": sample.instruction,
            "label": label,
            "reason": str(parsed.get("reason", "")).strip(),
            "annotator": "qwen3.5",
            "raw_output": output_text,
        }
        if sample.sample_id is not None:
            result["sample_id"] = sample.sample_id
        if sample.metadata is not None:
            result["metadata"] = sample.metadata
        return result


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"failed to parse JSON from model output: {text}")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError(f"expected JSON object, got: {parsed}")
    return parsed


def load_samples(path: Path) -> list[Sample]:
    if path.suffix.lower() == ".jsonl":
        items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        items = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(items, list):
        raise ValueError("input annotation manifest must be a list or jsonl sequence of objects")

    samples: list[Sample] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"sample at index {idx} is not an object")
        samples.append(
            Sample(
                prev_frame=str(item["prev_frame"]),
                frame=str(item["frame"]),
                instruction=str(item["instruction"]),
                sample_id=str(item["sample_id"]) if "sample_id" in item else None,
                metadata=item.get("metadata"),
            )
        )
    return samples


def write_results(path: Path, results: list[dict[str, Any]], fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        lines = [json.dumps(item, ensure_ascii=False) for item in results]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate frame-pair event labels with Qwen3.5.")
    parser.add_argument("--input", required=True, help="Path to unlabeled manifest (.json or .jsonl).")
    parser.add_argument("--output", required=True, help="Path to labeled output (.json or .jsonl).")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-9B",
        help="Hugging Face model id or local model path.",
    )
    parser.add_argument(
        "--cache-dir",
        default="models/hf",
        help="Local Hugging Face cache directory.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Model device. Use cpu or cuda.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap for debugging. 0 means all samples.",
    )
    parser.add_argument(
        "--format",
        choices=["json", "jsonl"],
        default="json",
        help="Output format.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    cache_dir = str((ROOT / args.cache_dir).resolve()) if not Path(args.cache_dir).is_absolute() else args.cache_dir

    samples = load_samples(input_path)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    annotator = QwenVLAnnotator(
        model_name_or_path=args.model,
        cache_dir=cache_dir,
        device=args.device,
    )

    results: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples, start=1):
        result = annotator.annotate(sample)
        results.append(result)
        print(f"[{idx}/{len(samples)}] label={result['label']} sample={result.get('sample_id', idx)}")

    write_results(output_path, results, fmt=args.format)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
