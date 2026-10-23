"""Callbacks Stable-Baselines3 : journalisation des récompenses et pause du vrai jeu."""
from __future__ import annotations

from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class RewardBreakdownCallback(BaseCallback):
    """Envoie dans TensorBoard le détail des récompenses par partie (points, pièces, étoiles...)."""

    def __init__(self, window: int = 50):
        super().__init__()
        self.hist: dict[str, deque] = {}
        self.window = window

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "intrinsic_reward" in info:
                self.logger.record_mean("curiosite/recompense_intrinseque", info["intrinsic_reward"])
            totals = info.get("episode_totals")
            if not totals:
                continue
            totals = {**totals, "score": info.get("score", 0), "pieces": info.get("coins", 0),
                      "etoiles": info.get("stars", 0)}
            for k, v in totals.items():
                self.hist.setdefault(k, deque(maxlen=self.window)).append(v)
                self.logger.record(f"partie/{k}", float(np.mean(self.hist[k])))
        return True


class PauseGameDuringUpdate(BaseCallback):
    """Le vrai jeu ne s'arrête pas pendant que le réseau apprend : on le met en pause."""

    def __init__(self, env):
        super().__init__()
        self.env = env

    def _on_rollout_end(self) -> None:
        self.env.pause()

    def _on_rollout_start(self) -> None:
        if self.num_timesteps > 0:
            self.env.resume()

    def _on_step(self) -> bool:
        return True


class LiveOverlayCallback(BaseCallback):
    """Affiche en temps réel les cadres autour des éléments découverts (premier environnement)."""

    def __init__(self, overlay):
        super().__init__()
        self.overlay = overlay

    def _on_step(self) -> bool:
        frame = self.training_env.env_method("frame_bgr", indices=[0])[0]
        obs = self.locals["new_obs"][:1]
        self.overlay.update(frame, self.locals["infos"][0], self.model, obs, self.locals["actions"][0])
        return True

    def _on_training_end(self) -> None:
        self.overlay.close()
