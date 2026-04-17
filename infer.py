from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.runner import WatcherRunner, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    runner = WatcherRunner(cfg, checkpoint=args.checkpoint, use_device=True)
    runner.run_forever()


if __name__ == "__main__":
    main()
