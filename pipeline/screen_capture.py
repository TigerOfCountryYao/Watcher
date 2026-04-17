import numpy as np
import cv2
import mss


class ScreenCapture:
    def __init__(self, monitor=1):
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[monitor]

    def get_frame(self):
        img = np.array(self.sct.grab(self.monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return frame