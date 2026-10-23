"""Environnement Gymnasium pour le VRAI Subway Surfers (capture d'écran + commandes).

Observation : l'image du jeu réduite (84x84 couleur par défaut). Aucune
information sur les objets n'est donnée à l'agent : c'est le réseau de
neurones convolutif qui apprend seul à les reconnaître.
"""
from __future__ import annotations

import time

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .controls import ScreenCapture, make_controller
from .hud import HudReader, TemplateDetector
from .restart import Restarter
from .rewards import RewardConfig, RewardTracker


class SubwayScreenEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        g = cfg["game"]
        self.size = g.get("obs_size", 84)
        self.grayscale = g.get("grayscale", False)
        self.step_interval = g.get("step_interval", 0.1)
        self.max_steps = g.get("max_episode_steps", 20000)
        self.n_actions = 6 if g.get("use_hoverboard", False) else 5

        self.capture = ScreenCapture(g["capture"])
        self.controller = make_controller(g)
        self.hud = HudReader(g)
        self.restarter = Restarter(g, self.controller, self.capture, self.hud)
        self.tracker = RewardTracker(RewardConfig.from_dict(cfg.get("rewards")))

        ch = 1 if self.grayscale else 3
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(0, 255, (self.size, self.size, ch), np.uint8)
        self._last_frame = None
        self._t_last = time.time()

    # ------------------------------------------------------------------
    def _to_obs(self, frame: np.ndarray) -> np.ndarray:
        top, bottom = self.cfg["game"].get("obs_crop_y", [0, 1.0])
        h = frame.shape[0]
        frame = frame[int(top * h):int(bottom * h)]  # on peut retirer le HUD de l'observation
        img = cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:, :, None]
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _wait_tick(self):
        dt = self.step_interval - (time.time() - self._t_last)
        if dt > 0:
            time.sleep(dt)
        self._t_last = time.time()

    def _click_template(self, name) -> bool:
        img = self.capture.grab()
        for t in self.cfg["game"].get("templates", {}).get(name) or []:
            try:
                pos = TemplateDetector(t["path"], t.get("region"), t.get("threshold", 0.8)).locate(img)
            except FileNotFoundError:
                continue
            if pos is not None:
                self.controller.click(*pos)
                return True
        return False

    def pause(self):
        # le bouton pause (image « ingame ») sert à mettre en pause ; sinon pause_sequence
        if not self._click_template("ingame"):
            self.controller.run_sequence(self.cfg["game"].get("pause_sequence"))

    def resume(self):
        time.sleep(0.5)
        if self._click_template("reprendre"):
            time.sleep(self.cfg["game"].get("resume_wait", 3.0))  # compte à rebours
        else:
            self.controller.run_sequence(self.cfg["game"].get("resume_sequence"))
        self._t_last = time.time()

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.restarter.restart()  # reconnaît les boutons à l'écran et relance une partie
        self.hud.reset()
        frame = self.capture.grab()
        # on attend que la partie ait vraiment commencé (plus d'écran de mort)
        for _ in range(100):
            state = self.hud.read(frame)
            if not state.dead:
                break
            time.sleep(0.1)
            frame = self.capture.grab()
        self.tracker.reset(state)
        self.t = 0
        self._t_last = time.time()
        self._last_frame = frame
        return self._to_obs(frame), {}

    def step(self, action):
        self.controller.act(action)
        self._wait_tick()
        frame = self.capture.grab()
        self._last_frame = frame
        state = self.hud.read(frame)
        reward, parts = self.tracker.step(state)
        self.t += 1
        terminated = state.dead
        truncated = self.t >= self.max_steps
        info = {"score": state.score, "coins": state.coins, "stars": state.stars,
                "multiplier": state.multiplier, "stumbled": state.stumbled, "reward_parts": parts}
        if terminated or truncated:
            info["episode_totals"] = dict(self.tracker.totals)
        return self._to_obs(frame), reward, terminated, truncated, info

    def render(self):
        return None if self._last_frame is None else cv2.cvtColor(self._last_frame, cv2.COLOR_BGR2RGB)

    def frame_bgr(self):
        """Dernière capture en pleine résolution (pour l'affichage des cadres)."""
        return self._last_frame
