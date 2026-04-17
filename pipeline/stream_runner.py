import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.model import ExcitementModel
from pipeline.screen_capture import ScreenCapture
from utils.preprocess import preprocess_frame


def main() -> None:
    cap = ScreenCapture()
    model = ExcitementModel()
    model.eval()

    prev_z = None
    instruction = torch.tensor([0], dtype=torch.long)

    while True:
        frame = cap.get_frame()
        frame_t = preprocess_frame(frame, image_size=224).unsqueeze(0)

        with torch.no_grad():
            z, excite = model(frame_t, instruction, prev_z)

        if excite is not None:
            score = excite.item()
            print(f"excite: {score:.4f}")
            if score > 0.5:
                print("TRIGGER EVENT")

        prev_z = z
        time.sleep(0.03)


if __name__ == "__main__":
    main()
