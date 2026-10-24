"""Petit détecteur du personnage, entraîné sur VOS images (calibrate.py player + train_player.py).

Le suivi par ressemblance d'image décroche trop souvent dans Subway Surfers City (effets
lumineux, hoverboard, sauts, bulle...). Un petit réseau convolutif entraîné sur une trentaine
d'images où le personnage a été encadré le retrouve bien mieux. Il prédit une « carte de
chaleur » du centre du personnage et la taille de son cadre.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IN_W, IN_H = 120, 256   # image réduite donnée au réseau
STRIDE = 4              # la carte de sortie est 4 fois plus petite


def _block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True))


class PlayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            _block(3, 24, 2), _block(24, 24),
            _block(24, 48, 2), _block(48, 48),
            _block(48, 64, 2), _block(64, 64),          # 1/8
        )
        self.ctx = nn.Sequential(_block(64, 64), nn.Conv2d(64, 64, 3, 1, 2, dilation=2), nn.ReLU(True))
        self.head = nn.Sequential(_block(64 + 48, 48), nn.Conv2d(48, 3, 1))  # chaleur + largeur + hauteur

    def forward(self, x):
        f2 = self.body[:4](x)           # 1/4
        f3 = self.ctx(self.body[4:](f2))
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(torch.cat([f3, f2], 1))


def _prep(img_bgr):
    x = cv2.resize(img_bgr, (IN_W, IN_H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return torch.from_numpy(x.transpose(2, 0, 1))


def _augment(img, box, rng):
    """Variations d'éclairage, de couleur, de position et d'échelle (le jeu change beaucoup)."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    s = rng.uniform(0.85, 1.15)
    tx, ty = rng.uniform(-0.12, 0.12) * w, rng.uniform(-0.1, 0.1) * h
    M = np.float32([[s, 0, (1 - s) * w / 2 + tx], [0, s, (1 - s) * h / 2 + ty]])
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    pts = np.array([[x0, y0], [x1, y1]], np.float32) @ M[:, :2].T + M[:, 2]
    x0, y0, x1, y1 = pts.ravel()
    if rng.random() < 0.5:
        img, x0, x1 = img[:, ::-1].copy(), w - x1, w - x0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-12, 12)) % 180
    hsv[..., 1] *= rng.uniform(0.7, 1.3)
    hsv[..., 2] = hsv[..., 2] * rng.uniform(0.6, 1.4) + rng.uniform(-20, 20)
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (5, 5), 0)
    return img, (x0, y0, x1, y1)


def _target(box, img_w, img_h):
    hw, hh = IN_W // STRIDE, IN_H // STRIDE
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2 / img_w * hw, (y0 + y1) / 2 / img_h * hh
    ys, xs = np.mgrid[0:hh, 0:hw]
    heat = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * 1.5 ** 2)).astype(np.float32)
    size = np.array([(x1 - x0) / img_w, (y1 - y0) / img_h], np.float32)
    return heat, size, (min(hw - 1, max(0, int(cx))), min(hh - 1, max(0, int(cy))))


def train(images, boxes, steps=1500, batch=16, lr=2e-3, seed=0, log=print, device="auto"):
    """images : liste d'images BGR (taille quelconque) ; boxes : (x0, y0, x1, y1) en pixels."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = PlayerNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    net.train()
    for step in range(steps):
        xs, heats, sizes, cells = [], [], [], []
        for _ in range(batch):
            k = rng.integers(len(images))
            img, box = _augment(images[k], boxes[k], rng)
            heat, size, cell = _target(box, img.shape[1], img.shape[0])
            xs.append(_prep(img))
            heats.append(heat)
            sizes.append(size)
            cells.append(cell)
        out = net(torch.stack(xs).to(device))
        heat_t = torch.from_numpy(np.stack(heats)).to(device)
        p = torch.sigmoid(out[:, 0])
        # perte « focale » de CenterNet : la plupart des pixels ne sont pas le personnage
        pos = heat_t > 0.99
        l_pos = -((1 - p) ** 2 * torch.log(p + 1e-6))[pos].sum()
        l_neg = -((1 - heat_t) ** 4 * p ** 2 * torch.log(1 - p + 1e-6))[~pos].sum()
        loss_h = (l_pos + l_neg) / max(1, int(pos.sum()))
        idx = torch.arange(len(cells), device=device)
        cx = torch.tensor([c[0] for c in cells], device=device)
        cy = torch.tensor([c[1] for c in cells], device=device)
        loss_s = F.l1_loss(torch.sigmoid(out[idx, 1:, cy, cx]), torch.from_numpy(np.stack(sizes)).to(device))
        loss = loss_h + 5 * loss_s
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if log and (step % 200 == 0 or step == steps - 1):
            log(f"  étape {step:5d}/{steps}  perte {loss.item():.3f}")
    net.eval()
    return net.cpu()


class PlayerDetector:
    def __init__(self, path="models/player_detector.pt", min_conf=0.1):
        self.net = PlayerNet()
        self.net.load_state_dict(torch.load(path, map_location="cpu"))
        self.net.eval()
        self.min_conf = min_conf
        self.last_conf = 0.0

    @torch.no_grad()
    def find(self, frame_bgr):
        """Renvoie le cadre (x, y, w, h) du personnage en pixels, ou None s'il n'est pas vu."""
        h, w = frame_bgr.shape[:2]
        out = self.net(_prep(frame_bgr)[None])[0]
        heat = torch.sigmoid(out[0])
        i = int(heat.argmax())
        cy, cx = divmod(i, heat.shape[1])
        self.last_conf = float(heat[cy, cx])
        if self.last_conf < self.min_conf:
            return None
        bw, bh = torch.sigmoid(out[1:, cy, cx]).tolist()
        px, py = (cx + 0.5) / heat.shape[1] * w, (cy + 0.5) / heat.shape[0] * h
        return int(px - bw * w / 2), int(py - bh * h / 2), int(bw * w), int(bh * h)


def save(net, path="models/player_detector.pt"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(net.state_dict(), path)


def load_labels(folder="calib/player"):
    """Images + cadres enregistrés par « calibrate.py player »."""
    meta = os.path.join(folder, "labels.json")
    if not os.path.exists(meta):
        return [], []
    with open(meta, encoding="utf-8") as f:
        labels = json.load(f)
    imgs, boxes = [], []
    for fn, box in labels.items():
        img = cv2.imread(os.path.join(folder, fn))
        if img is not None:
            imgs.append(img)
            boxes.append(tuple(box))
    return imgs, boxes


def head_crop(frame_bgr, box, size=(128, 96)):
    """Zone autour de la tête du personnage (les étoiles d'étourdissement tournent là)."""
    x, y, w, h = box
    cx = x + w // 2
    y0, x0 = max(0, y - int(0.35 * h)), max(0, cx - int(0.9 * w))
    crop = frame_bgr[y0:y0 + int(0.75 * h), x0:x0 + int(1.8 * w)]
    if crop.size == 0:
        crop = frame_bgr
    return cv2.resize(crop, size)
