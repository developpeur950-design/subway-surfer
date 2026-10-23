"""Mini-simulateur inspiré de Subway Surfers, pour tester l'apprentissage sans le vrai jeu.

L'agent ne reçoit QUE des pixels (vue de dessus 84x84 en couleur) : il ne sait pas
ce qu'est une pièce, un train ou une barrière. Il doit découvrir tout seul, grâce
aux récompenses, ce que signifie chaque forme/couleur.

Éléments du monde (3 voies) :
  - pièce            -> +pièce
  - étoile           -> +1 au multiplicateur de la partie (simplifié)
  - barrière basse   -> il faut sauter, sinon mort
  - barrière haute   -> il faut rouler, sinon mort
  - train            -> il faut changer de voie ; foncer dans l'avant = mort,
                        se cogner sur le côté en changeant de voie = trébuchement.
                        Deux trébuchements rapprochés = attrapé par l'inspecteur (mort).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .rewards import HudState, RewardConfig, RewardTracker

EMPTY, COIN, STAR, LOW, HIGH, TRAIN = range(6)
ACTIONS = ["rien", "gauche", "droite", "saut", "roulade"]

COLORS = {
    COIN: (255, 215, 0),
    STAR: (200, 60, 255),
    LOW: (230, 70, 40),
    HIGH: (40, 110, 240),
    TRAIN: (150, 150, 150),
}


class SubwaySimEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}

    def __init__(self, reward_config: dict | None = None, depth: int = 20, size: int = 84,
                 max_steps: int = 5000, render_mode: str | None = "rgb_array"):
        super().__init__()
        self.depth, self.size, self.max_steps = depth, size, max_steps
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(0, 255, (size, size, 3), np.uint8)
        self.tracker = RewardTracker(RewardConfig.from_dict(reward_config))

    # ------------------------------------------------------------------ monde
    def _spawn_row(self) -> np.ndarray:
        row = np.full(3, EMPTY, np.int8)
        rng = self.np_random
        if self._train_left.any():
            row[self._train_left > 0] = TRAIN
            self._train_left = np.maximum(self._train_left - 1, 0)
        if self._gap > 0:
            self._gap -= 1
        else:
            r = rng.random()
            free = np.flatnonzero(row == EMPTY)
            if r < 0.30 and len(free) > 1:
                # obstacles : on garde toujours au moins une voie praticable
                n = rng.integers(1, len(free))
                for lane in rng.choice(free, n, replace=False):
                    kind = rng.choice([LOW, HIGH, TRAIN], p=[0.35, 0.35, 0.30])
                    row[lane] = kind
                    if kind == TRAIN:
                        self._train_left[lane] = rng.integers(4, 10)
                self._gap = 3
            elif r < 0.33:
                row[rng.choice(free)] = STAR
            elif r < 0.60:
                lane = rng.choice(free)
                row[lane] = COIN
        # pièces en ligne
        if self._coin_lane >= 0 and row[self._coin_lane] == EMPTY:
            row[self._coin_lane] = COIN
            self._coin_run -= 1
            if self._coin_run <= 0:
                self._coin_lane = -1
        elif self.np_random.random() < 0.05:
            self._coin_lane, self._coin_run = int(self.np_random.integers(3)), int(self.np_random.integers(4, 10))
        return row

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.full((self.depth, 3), EMPTY, np.int8)  # ligne 0 = au niveau du joueur
        self._train_left = np.zeros(3, np.int32)
        self._gap, self._coin_lane, self._coin_run = 6, -1, 0
        for i in range(8, self.depth):
            self.grid[i] = self._spawn_row()
        self.lane, self.air, self.roll = 1, 0, 0
        self.score, self.coins, self.stars = 0, 0, 0
        self.mult_timer, self.stumble_timer, self.t = 0, 0, 0
        self.hud = HudState()
        self.tracker.reset(self.hud)
        return self._obs(), {}

    def step(self, action: int):
        self.t += 1
        stumbled = dead = False

        # --- action du joueur
        if action in (1, 2):
            target = self.lane + (-1 if action == 1 else 1)
            if 0 <= target <= 2:
                if self.grid[0, target] == TRAIN or self.grid[1, target] == TRAIN:
                    stumbled = True  # on se cogne contre le flanc d'un train
                else:
                    self.lane = target
        elif action == 3 and self.air == 0:
            self.air, self.roll = 4, 0
        elif action == 4:
            self.air, self.roll = 0, 3

        # --- le monde avance d'une ligne
        self.grid[:-1] = self.grid[1:]
        self.grid[-1] = self._spawn_row()
        cell = self.grid[0, self.lane]
        if cell == COIN:
            self.coins += 1
            self.grid[0, self.lane] = EMPTY
        elif cell == STAR:
            self.stars += 1
            self.mult_timer = 15  # petit flash à l'écran
            self.grid[0, self.lane] = EMPTY
        elif cell == LOW and self.air == 0:
            dead = True
        elif cell == HIGH and self.roll == 0:
            dead = True
        elif cell == TRAIN:
            dead = True

        if stumbled:
            if self.stumble_timer > 0:
                dead = True  # l'inspecteur nous rattrape
            self.stumble_timer = 40
        self.stumble_timer = max(0, self.stumble_timer - 1)
        self.air, self.roll = max(0, self.air - 1), max(0, self.roll - 1)
        self.mult_timer = max(0, self.mult_timer - 1)

        multiplier = 1 + self.stars
        if not dead:
            self.score += multiplier
        self.hud = HudState(self.score, self.coins, multiplier, self.stars, dead, stumbled)
        reward, parts = self.tracker.step(self.hud)
        truncated = self.t >= self.max_steps
        info = {"score": self.score, "coins": self.coins, "stars": self.stars,
                "stumbled": stumbled, "reward_parts": parts}
        if dead or truncated:
            info["episode_totals"] = dict(self.tracker.totals)
        return self._obs(), reward, dead, truncated, info

    # ------------------------------------------------------------------ rendu
    def _obs(self) -> np.ndarray:
        s = self.size
        img = np.full((s, s, 3), 30, np.uint8)
        lane_w = s // 3
        row_h = s / (self.depth + 2)
        img[:, lane_w - 1:lane_w + 1] = 60
        img[:, 2 * lane_w - 1:2 * lane_w + 1] = 60

        def y_of(i):  # ligne 0 en bas de l'image
            y1 = int(s - (i + 2) * row_h)
            return max(0, y1), int(y1 + row_h)

        for i in range(self.depth):
            for lane in range(3):
                c = self.grid[i, lane]
                if c == EMPTY:
                    continue
                y0, y1 = y_of(i)
                x0, x1 = lane * lane_w + 3, (lane + 1) * lane_w - 3
                if c in (COIN, STAR):
                    x0, x1 = x0 + 7, x1 - 7
                elif c == LOW:
                    y0 = y1 - 2
                img[y0:y1, x0:x1] = COLORS[c]

        # joueur (plus large en l'air, plus petit en roulade)
        y0, y1 = y_of(0)
        pad = 4 if self.air else (11 if self.roll else 8)
        x0, x1 = self.lane * lane_w + pad, (self.lane + 1) * lane_w - pad
        img[max(0, y0 - 3):s, x0:x1] = (60, 220, 90)
        # barre de multiplicateur / alerte inspecteur, comme le HUD du jeu
        if self.mult_timer:
            img[0:3, :] = COLORS[STAR]
        if self.stumble_timer:
            img[s - 3:s, :] = (255, 0, 0)
        return img

    def render(self):
        return self._obs()
