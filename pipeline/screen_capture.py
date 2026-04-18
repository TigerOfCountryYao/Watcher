from __future__ import annotations

import ctypes
from ctypes import wintypes

import cv2
import mss
import numpy as np


DWMWA_EXTENDED_FRAME_BOUNDS = 9
SW_RESTORE = 9


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class ScreenCapture:
    def __init__(
        self,
        mode: str = "desktop",
        monitor: int = 1,
        window_title: str | None = None,
    ) -> None:
        self.mode = mode
        self.monitor = monitor
        self.window_title = window_title
        self.sct = mss.mss()

        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.dwmapi = ctypes.WinDLL("dwmapi", use_last_error=True)
        self.user32.SetProcessDPIAware()

        self.user32.EnumWindows.argtypes = [
            ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM),
            wintypes.LPARAM,
        ]
        self.user32.IsWindowVisible.argtypes = [wintypes.HWND]
        self.user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        self.user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        self.user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        self.user32.SetForegroundWindow.argtypes = [wintypes.HWND]
        self.dwmapi.DwmGetWindowAttribute.argtypes = [
            wintypes.HWND,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        self.last_window_title: str | None = None
        self.last_region: dict[str, int] | None = None

    def get_frame(self) -> np.ndarray:
        if self.mode == "desktop":
            region = self.sct.monitors[0]
            self.last_window_title = None
        elif self.mode == "monitor":
            region = self.sct.monitors[self.monitor]
            self.last_window_title = None
        elif self.mode == "window":
            hwnd = self._find_window(self.window_title)
            if hwnd is None:
                raise RuntimeError(f"window not found: {self.window_title}")
            self.user32.ShowWindow(hwnd, SW_RESTORE)
            self.user32.SetForegroundWindow(hwnd)
            region = self._get_window_region(hwnd)
            self.last_window_title = self._get_window_text(hwnd)
        else:
            raise ValueError(f"unsupported capture mode: {self.mode}")

        self.last_region = dict(region)
        img = np.array(self.sct.grab(region))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def _find_window(self, title_pattern: str | None) -> int | None:
        if not title_pattern:
            raise ValueError("window_title is required when mode='window'")

        patterns = [part.strip() for part in title_pattern.split("|") if part.strip()]
        matches: list[int] = []

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def enum_proc(hwnd: int, lparam: int) -> bool:
            if not self.user32.IsWindowVisible(hwnd):
                return True

            length = self.user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return True

            buffer = ctypes.create_unicode_buffer(length + 1)
            self.user32.GetWindowTextW(hwnd, buffer, length + 1)
            title = buffer.value
            if any(pattern.lower() in title.lower() for pattern in patterns):
                matches.append(hwnd)
                return False
            return True

        self.user32.EnumWindows(enum_proc, 0)
        return matches[0] if matches else None

    def _get_window_text(self, hwnd: int) -> str:
        length = self.user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return ""
        buffer = ctypes.create_unicode_buffer(length + 1)
        self.user32.GetWindowTextW(hwnd, buffer, length + 1)
        return buffer.value

    def _get_window_region(self, hwnd: int) -> dict[str, int]:
        rect = RECT()
        result = self.dwmapi.DwmGetWindowAttribute(
            hwnd,
            DWMWA_EXTENDED_FRAME_BOUNDS,
            ctypes.byref(rect),
            ctypes.sizeof(rect),
        )
        if result != 0:
            raise OSError(f"DwmGetWindowAttribute failed with code {result}")

        width = max(1, rect.right - rect.left)
        height = max(1, rect.bottom - rect.top)
        return {
            "left": rect.left,
            "top": rect.top,
            "width": width,
            "height": height,
        }
