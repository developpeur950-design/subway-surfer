"""Capture d'écran et envoi des commandes au vrai jeu.

- ``KeyboardController`` : version navigateur/PC (flèches du clavier).
- ``MouseSwipeController`` : Android affiché sur le PC (scrcpy, émulateur) via des swipes à la souris.
- ``AdbController``      : Android via des swipes adb (plus simple mais plus lent).
"""
from __future__ import annotations

import subprocess
import sys
import time

import numpy as np

if sys.platform == "win32":
    # Sous Windows avec une mise à l'échelle (125 %, 150 %...), la capture et la
    # souris n'utiliseraient pas les mêmes coordonnées : on force les pixels réels.
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass

# 0 rien, 1 gauche, 2 droite, 3 saut, 4 roulade, 5 hoverboard (optionnel)
ACTION_NAMES = ["rien", "gauche", "droite", "saut", "roulade", "hoverboard"]


class ScreenCapture:
    def __init__(self, region: dict):
        import mss
        self.sct = mss.mss()
        self.region = {k: int(region[k]) for k in ("left", "top", "width", "height")}

    def grab(self) -> np.ndarray:
        """Renvoie l'image BGR de la zone du jeu."""
        return np.asarray(self.sct.grab(self.region))[:, :, :3].copy()


class KeyboardController:
    KEYS = {1: "left", 2: "right", 3: "up", 4: "down", 5: "space"}

    def __init__(self, cfg: dict):
        import pyautogui
        pyautogui.PAUSE = 0
        pyautogui.FAILSAFE = True  # souris dans un coin de l'écran = arrêt d'urgence
        self.pg = pyautogui
        self.region = cfg["capture"]

    def act(self, action: int):
        key = self.KEYS.get(int(action))
        if key == "space":
            self.pg.press("space", presses=2, interval=0.05)
        elif key:
            self.pg.press(key)

    def run_sequence(self, steps):
        """Étapes : {"key": "space"}, {"click": [x, y]} (coord. dans la zone du jeu), {"wait": 1.5}."""
        for st in steps or []:
            if "key" in st:
                self.pg.press(st["key"])
            if "click" in st:
                x, y = st["click"]
                self.pg.click(self.region["left"] + x, self.region["top"] + y)
            if "wait" in st:
                time.sleep(st["wait"])


class MouseSwipeController(KeyboardController):
    """Swipes à la souris dans une fenêtre qui recopie le téléphone (scrcpy, émulateur...).

    scrcpy transforme les glisser-déposer de la souris en vrais gestes tactiles,
    avec beaucoup moins de latence que "adb shell input swipe".
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        m = cfg.get("mouse_swipe", {})
        self.dist = m.get("distance", 0.25)      # longueur du swipe (fraction de la zone)
        self.duration = m.get("duration", 0.05)  # secondes
        self.cy = m.get("start_y", 0.6)          # hauteur de départ (fraction de la zone)

    def _swipe(self, dx: float, dy: float):
        r = self.region
        x = r["left"] + r["width"] // 2
        y = r["top"] + int(r["height"] * self.cy)
        d = self.dist * min(r["width"], r["height"])
        self.pg.moveTo(x, y)
        self.pg.dragTo(x + dx * d, y + dy * d, duration=self.duration, button="left")

    def act(self, action: int):
        action = int(action)
        moves = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
        if action in moves:
            self._swipe(*moves[action])
        elif action == 5:
            r = self.region
            self.pg.click(r["left"] + r["width"] // 2, r["top"] + r["height"] // 2, clicks=2, interval=0.05)


class AdbController:
    def __init__(self, cfg: dict):
        a = cfg.get("adb", {})
        self.serial = a.get("serial")
        self.w, self.h = a.get("device_width", 1080), a.get("device_height", 2400)
        self.duration = a.get("swipe_ms", 60)
        cmd = ["adb"] + (["-s", self.serial] if self.serial else []) + ["shell"]
        # un shell adb persistant : bien plus rapide qu'un processus par geste
        self.shell = subprocess.Popen(cmd, stdin=subprocess.PIPE, text=True)

    def _send(self, line: str):
        self.shell.stdin.write(line + "\n")
        self.shell.stdin.flush()

    def _swipe(self, dx: float, dy: float):
        cx, cy = self.w // 2, int(self.h * 0.6)
        ex, ey = int(cx + dx * self.w * 0.3), int(cy + dy * self.h * 0.15)
        self._send(f"input swipe {cx} {cy} {ex} {ey} {self.duration}")

    def act(self, action: int):
        action = int(action)
        if action == 1:
            self._swipe(-1, 0)
        elif action == 2:
            self._swipe(1, 0)
        elif action == 3:
            self._swipe(0, -1)
        elif action == 4:
            self._swipe(0, 1)
        elif action == 5:
            self._send(f"input tap {self.w // 2} {self.h // 2}")
            self._send(f"input tap {self.w // 2} {self.h // 2}")

    def run_sequence(self, steps):
        """Étapes : {"tap": [x, y]} (pixels de l'appareil), {"key": "BACK"}, {"wait": 1.5}."""
        for st in steps or []:
            if "tap" in st:
                self._send("input tap {} {}".format(*st["tap"]))
            if "key" in st:
                self._send(f"input keyevent {st['key']}")
            if "wait" in st:
                time.sleep(st["wait"])


def make_controller(cfg: dict):
    kind = cfg.get("controller", "keyboard")
    if kind == "keyboard":
        return KeyboardController(cfg)
    if kind == "mouse_swipe":
        return MouseSwipeController(cfg)
    if kind == "adb":
        return AdbController(cfg)
    raise ValueError(f"controller inconnu : {kind}")
