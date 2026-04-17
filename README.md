# Watcher

Real-time task-conditioned visual event detection for screen agents.

Watcher is a lightweight perception layer for screen-based agents. Instead of
pixel-diff or polling, it predicts whether the current frame contains a
task-relevant visual event.

## Instruction

Watcher uses one unified `instruction` concept everywhere:

- training data: each sample contains an `instruction` string
- annotation input: each sample contains an `instruction` string
- online inference: `infer.instruction` in `configs/config.yaml`
- model forward: the model receives a list of instruction strings

The instruction should describe one concrete visual event or target state in
plain language. It should be short, specific, and directly observable on the
screen.

Recommended examples:

- `watch for a popup appearing`
- `watch whether the submit button becomes clickable`
- `watch for the selected chat item changing in WeChat`
- `watch for a desktop context menu appearing`

Avoid:

- `watch for something important`
- `click the submit button`
- multiple conditions in one instruction
- hidden intent that is not visible on screen

## Core Idea

We model perception as:

`f(prev_frame, frame, instruction) -> event probability`

Instead of asking "what changed?", Watcher asks:
"is this change relevant to the current task?"

## Architecture

`Prev frame + Curr frame + Instruction -> Backbone -> Feature diff -> FiLM -> MLP -> event`

- Backbone: extracts visual feature maps.
- Text encoder: converts the instruction string into an embedding.
- Feature diff: compares current and previous feature maps.
- FiLM: conditions change features by the instruction embedding.
- Head: predicts a binary event logit.

## Project Structure

```text
Watcher/
  model/
    backbone.py
    film.py
    excite_head.py
    text_encoder.py
    model.py
  pipeline/
    screen_capture.py
    stream_runner.py
  data/
    dataset.py
    annotate_qwenvl.py
    unlabeled.json
  train/
    train.py
  utils/
    preprocess.py
    logger.py
  configs/
    config.yaml
  infer.py
  requirements.txt
  README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Run realtime stream:

```bash
python pipeline/stream_runner.py
```

Run training:

```bash
python train/train.py --config configs/config.yaml
```

Run inference:

```bash
python infer.py --config configs/config.yaml
```

Inference uses:

- `infer.capture_mode`
- `infer.window_title`
- `infer.instruction`

## Data Format

Labeled training data:

```json
[
  {
    "prev_frame": "path/to/prev.png",
    "frame": "path/to/current.png",
    "instruction": "watch for a popup appearing",
    "label": 1
  }
]
```

`label` semantics:

- `1`: the current frame contains a task-relevant event for the instruction
- `0`: the current frame does not contain a task-relevant event for the instruction

Unlabeled manifest for Qwen annotation:

```json
[
  {
    "sample_id": "desktop-0003",
    "prev_frame": "data/raw/frame_pairs/desktop-0003_prev.png",
    "frame": "data/raw/frame_pairs/desktop-0003_curr.png",
    "instruction": "watch for a desktop context menu appearing",
    "metadata": {
      "scene": "desktop",
      "capture_type": "fullscreen"
    }
  }
]
```

Generate labels with Qwen3.5:

```bash
python data/annotate_qwenvl.py ^
  --input data/unlabeled.json ^
  --output data/labeled.json ^
  --model Qwen/Qwen3.5-9B ^
  --cache-dir models/hf ^
  --device cuda
```

## Capture Modes

Watcher uses one capture module with three modes:

- `desktop`: capture the full desktop
- `monitor`: capture one monitor by index
- `window`: capture one application window using its real visible bounds

Example window-mode inference:

```yaml
infer:
  capture_mode: "window"
  window_title: "微信"
  instruction: "watch for the selected chat item changing in WeChat"
```

## License

MIT
