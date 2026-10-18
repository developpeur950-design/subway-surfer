"""Curiosité par Random Network Distillation (RND, Burda et al. 2018).

Un réseau "cible" aléatoire et figé encode chaque image ; un réseau "prédicteur"
apprend à imiter cette sortie. Sur les images déjà vues souvent, l'erreur de
prédiction est faible ; sur les situations nouvelles elle est forte. Cette erreur
est ajoutée comme récompense intrinsèque : l'agent est poussé à DÉCOUVRIR de
nouvelles situations (nouveaux obstacles, power-ups, toits des trains...)
au lieu de répéter toujours la même chose.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import VecEnvWrapper


def _encoder(in_ch: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_ch, 32, 8, 4), nn.LeakyReLU(),
        nn.Conv2d(32, 64, 4, 2), nn.LeakyReLU(),
        nn.Conv2d(64, 64, 3, 1), nn.LeakyReLU(),
        nn.Flatten(), nn.LazyLinear(out_dim),
    )


class VecRND(VecEnvWrapper):
    def __init__(self, venv, frame_channels: int, beta: float = 0.5, lr: float = 1e-4,
                 update_every: int = 256, epochs: int = 2, batch_size: int = 64,
                 device: str = "auto"):
        super().__init__(venv)
        self.frame_channels, self.beta = frame_channels, beta
        self.update_every, self.epochs, self.batch_size = update_every, epochs, batch_size
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                                   else ("cpu" if device == "auto" else device))
        self.target = _encoder(1, 256).to(self.device)
        self.predictor = nn.Sequential(_encoder(1, 256), nn.ReLU(),
                                       nn.Linear(256, 256)).to(self.device)
        h, w = self.observation_space.shape[:2]
        with torch.no_grad():  # initialise les couches "lazy"
            dummy = torch.zeros(1, 1, h, w, device=self.device)
            self.target(dummy), self.predictor(dummy)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.obs_rms = RunningMeanStd(shape=(1, h, w))
        self.int_rms = RunningMeanStd(shape=())
        self.buffer: list[np.ndarray] = []
        self.last_intrinsic = 0.0

    def _last_frame(self, obs: np.ndarray) -> np.ndarray:
        # obs : (n, H, W, C*k) après VecFrameStack -> dernière image en niveaux de gris (n, 1, H, W)
        last = obs[..., -self.frame_channels:].astype(np.float32).mean(axis=-1)
        return last[:, None]

    def _normalize(self, x: np.ndarray) -> torch.Tensor:
        x = (x - self.obs_rms.mean) / (np.sqrt(self.obs_rms.var) + 1e-8)
        return torch.as_tensor(np.clip(x, -5, 5), dtype=torch.float32, device=self.device)

    def _intrinsic(self, frames: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = self._normalize(frames)
            err = (self.predictor(x) - self.target(x)).pow(2).mean(dim=1)
        return err.cpu().numpy()

    def _train(self):
        data = np.concatenate(self.buffer)
        self.buffer.clear()
        self.obs_rms.update(data)
        for _ in range(self.epochs):
            idx = np.random.permutation(len(data))
            for i in range(0, len(data), self.batch_size):
                x = self._normalize(data[idx[i:i + self.batch_size]])
                loss = (self.predictor(x) - self.target(x)).pow(2).mean()
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

    def reset(self):
        obs = self.venv.reset()
        self.obs_rms.update(self._last_frame(obs))
        return obs

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        frames = self._last_frame(obs)
        r_int = self._intrinsic(frames)
        self.int_rms.update(r_int)
        r_int = r_int / (np.sqrt(self.int_rms.var) + 1e-8)
        self.last_intrinsic = float(r_int.mean())
        for info, ri in zip(infos, r_int):
            info["intrinsic_reward"] = float(ri)
        self.buffer.append(frames)
        if sum(len(b) for b in self.buffer) >= self.update_every:
            self._train()
        return obs, rewards + self.beta * r_int.astype(np.float32), dones, infos
