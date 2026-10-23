"""Petit classifieur « oui / non » sur une partie de l'écran, entraîné sur VOS images.

Exemple : « le garde est-il visible en bas de l'écran ? » (il apparaît après un choc contre
un train). Une image de référence fixe le reconnaît mal (il bouge, change de pose, le décor
change) ; un petit réseau entraîné sur quelques dizaines d'images étiquetées
(calibrate.py label <nom> puis train_classifier.py <nom>) est bien plus fiable.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IN_W, IN_H = 128, 96
# zone regardée par défaut, en fractions (x0, y0, x1, y1) : le bas de l'écran pour le garde
DEFAULT_ZONES = {"garde": (0.0, 0.55, 1.0, 1.0)}


def _crop(img, zone):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = zone
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def _prep(img, zone):
    x = cv2.resize(_crop(img, zone), (IN_W, IN_H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return torch.from_numpy(x.transpose(2, 0, 1))


class EventNet(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                                 nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(3, 16), blk(16, 32), blk(32, 48), blk(48, 64))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(F.adaptive_max_pool2d(self.f(x), 1).flatten(1)).squeeze(1)


def _augment(img, rng):
    h, w = img.shape[:2]
    s = rng.uniform(0.9, 1.1)
    M = np.float32([[s, 0, (1 - s) * w / 2 + rng.uniform(-0.06, 0.06) * w],
                    [0, s, (1 - s) * h / 2 + rng.uniform(-0.04, 0.04) * h]])
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    if rng.random() < 0.5:
        img = img[:, ::-1].copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-8, 8)) % 180
    hsv[..., 1] *= rng.uniform(0.75, 1.25)
    hsv[..., 2] = hsv[..., 2] * rng.uniform(0.65, 1.35) + rng.uniform(-15, 15)
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if rng.random() < 0.4:  # flou / basse résolution (vidéo, Wi-Fi...)
        f = rng.uniform(0.35, 0.8)
        img = cv2.resize(cv2.resize(img, None, fx=f, fy=f), (w, h))
    return img


def train(images, labels, zone, steps=800, batch=32, lr=2e-3, seed=0, log=print, device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    labels = np.asarray(labels, np.float32)
    pos, neg = np.flatnonzero(labels > 0.5), np.flatnonzero(labels <= 0.5)
    net = EventNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    net.train()
    for step in range(steps):
        # moitié d'exemples « oui », moitié « non » (les « oui » sont bien plus rares)
        idx = np.concatenate([rng.choice(pos, batch // 2), rng.choice(neg, batch - batch // 2)])
        x = torch.stack([_prep(_augment(_crop(images[i], zone), rng), (0, 0, 1, 1)) for i in idx]).to(device)
        y = torch.from_numpy(labels[idx]).to(device)
        loss = F.binary_cross_entropy_with_logits(net(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if log and (step % 200 == 0 or step == steps - 1):
            log(f"  étape {step:4d}/{steps}  perte {loss.item():.3f}")
    net.eval()
    return net.cpu()


class EventClassifier:
    def __init__(self, path, zone=None, threshold=0.5):
        d = torch.load(path, map_location="cpu")
        self.net = EventNet()
        self.net.load_state_dict(d["state"])
        self.net.eval()
        self.zone = tuple(zone or d.get("zone") or (0, 0, 1, 1))
        self.threshold = threshold
        self.last = 0.0

    @torch.no_grad()
    def prob(self, frame_bgr) -> float:
        self.last = float(torch.sigmoid(self.net(_prep(frame_bgr, self.zone)[None]))[0])
        return self.last

    def __call__(self, frame_bgr) -> bool:
        return self.prob(frame_bgr) >= self.threshold


def save(net, zone, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"state": net.state_dict(), "zone": list(zone)}, path)


def load_labels(name, folder="calib/labels"):
    """Images + étiquettes (1 = oui, 0 = non) enregistrées par « calibrate.py label <nom> »."""
    meta = os.path.join(folder, name, "labels.json")
    if not os.path.exists(meta):
        return [], []
    with open(meta, encoding="utf-8") as f:
        labels = json.load(f)
    imgs, ys = [], []
    for fn, y in labels.items():
        img = cv2.imread(os.path.join(folder, name, fn))
        if img is not None:
            imgs.append(img)
            ys.append(int(y))
    return imgs, ys
