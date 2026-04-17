import torch
import time
from pipeline.screen_capture import ScreenCapture
from model.model import ExcitementModel


def main():
    cap = ScreenCapture()
    model = ExcitementModel()
    model.eval()

    prev_z = None
    instruction = torch.tensor([0])  # dummy task id

    while True:
        frame = cap.get_frame()

        # preprocess
        frame_t = torch.from_numpy(frame).permute(2,0,1).unsqueeze(0).float() / 255.0

        with torch.no_grad():
            z, excite = model(frame_t, instruction, prev_z)

        if excite is not None:
            score = excite.item()
            print("excite:", score)

            if score > 0.5:
                print("⚡ TRIGGER EVENT")

        prev_z = z

        time.sleep(0.03)  # ~30fps


if __name__ == "__main__":
    main()