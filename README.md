# Watcher

Real-time task-conditioned visual excitement detection for screen agents.

Watcher is a lightweight perception layer for screen-based agents. Instead of
pixel-diff or polling, it predicts whether a visual change is relevant to the
current task.

## Core Idea

We model perception as:

`f(frame, instruction) -> excitement score`

Instead of asking "what changed?", Watcher asks:
"is this change relevant to the current task?"

## Architecture

`Frame -> Backbone -> FiLM -> z_t -> delta(z) -> MLP -> excitement`

- Backbone: extracts visual feature map.
- FiLM: conditions feature map by instruction embedding.
- Temporal delta: compares current and previous embedding.
- Head: predicts excitement score from delta features.

## Project Structure

```text
Watcher/
  model/
    backbone.py
    film.py
    excite_head.py
    model.py
  pipeline/
    screen_capture.py
    stream_runner.py
  data/
    dataset.py
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

Run training (uses random dataset when no annotation file is provided):

```bash
python train/train.py --config configs/config.yaml
```

Run inference entry:

```bash
python infer.py --config configs/config.yaml
```

## Data Format

If you want to train with your own data, set `data.annotation_file` in
`configs/config.yaml` to a JSON file like:

```json
[
  {
    "prev_frame": "path/to/prev.png",
    "frame": "path/to/current.png",
    "instruction_id": 3,
    "target": 0.87
  }
]
```

## License

MIT
