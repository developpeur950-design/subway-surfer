"""Affichage en temps réel : un cadre autour de chaque élément du jeu.

Deux choses sont superposées à l'image du jeu :

1. ``ElementDiscovery`` – découverte NON SUPERVISÉE des éléments. Personne ne lui dit ce
   qu'est un train ou une pièce : il repère ce qui se détache du décor (modèle de fond
   adaptatif), puis regroupe les éléments qui se ressemblent (couleur, forme, taille) en
   « types » qu'il invente lui-même (T0, T1, ...). Il apprend ensuite, pour chaque type,
   ce qui se passe après un contact avec le personnage, et en déduit seul si c'est un
   OBSTACLE (plus souvent suivi d'un choc ou d'une mort que la moyenne, cadre rouge) ou un
   BONUS (plus souvent suivi de pièces/étoiles dans les secondes qui suivent : pièces, mais
   aussi les bonus comme l'aimant, qui rapportent des pièces après coup, cadre vert).

2. L'attention de l'agent – une carte de chaleur montrant les pixels qui influencent le
   plus la décision du réseau de neurones qui joue (gradient de la politique).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

PALETTE = [(66, 135, 245), (60, 200, 90), (245, 66, 66), (240, 200, 40), (190, 80, 230),
           (40, 210, 220), (250, 140, 40), (230, 90, 170), (150, 150, 150), (120, 230, 160),
           (90, 90, 230), (200, 200, 90)]  # BGR
ACTION_NAMES = ["rien", "gauche", "droite", "saut", "roulade", "hoverboard"]


@dataclass
class Element:
    box: tuple[int, int, int, int]  # x, y, w, h (pixels de l'image du jeu)
    type_id: int
    attention: float = 0.0


class ElementDiscovery:
    def __init__(self, n_types: int = 10, work_width: int = 240, diff_threshold: float = 30,
                 min_area: float = 0.0008, history: int = 25, ignore_top: float = 0.0,
                 roi=None,
                 new_type_distance: float = 0.6, player_zone=(0.3, 0.55, 0.7, 0.85),
                 obstacle_factor: float = 1.5, bonus_factor: float = 1.5,
                 bonus_horizon: int = 12, danger_horizon: int = 5, min_touches: int = 10):
        self.n_types, self.work_width = n_types, work_width
        self.diff_threshold, self.min_area = diff_threshold, min_area
        self.obstacle_factor, self.bonus_factor = obstacle_factor, bonus_factor
        self.bonus_horizon, self.danger_horizon = bonus_horizon, danger_horizon
        self.min_touches = min_touches
        # fond = médiane des dernières images : un objet qui passe n'y laisse pas de "fantôme"
        self.frames: deque = deque(maxlen=history)
        self._t = 0
        self.ignore_top, self.new_type_distance = ignore_top, new_type_distance
        # zone de recherche (polygone en fractions de l'image) : les rails, pas le décor qui défile
        self.roi = np.array(roi, np.float32) if roi else None
        self._roi_mask = None
        self.player_zone = player_zone  # x0, y0, x1, y1 en fractions de l'image
        self.bg = None
        self.centroids: list[np.ndarray] = []
        self.counts: list[int] = []
        # Pour chaque type, après un contact avec le personnage :
        self.gain: list[float] = []    # gains (pièces, étoiles) dans les secondes qui suivent
        self.danger: list[float] = []  # pertes (choc, mort) juste après
        self.touches: list[int] = []
        self.avg_gain, self.avg_danger = 0.0, 0.0  # moyennes tous types confondus
        self._prev_here: set = set()
        self._open: list[list] = []  # contacts en cours d'évaluation : [type, âge, gain, danger]

    # ------------------------------------------------------------ détection
    def _descriptor(self, hsv_patch, mask_patch, w, h, cy, area_frac):
        pix = hsv_patch[mask_patch > 0]
        if len(pix) == 0:
            pix = hsv_patch.reshape(-1, 3)
        hue = np.histogram(pix[:, 0], bins=12, range=(0, 180), weights=pix[:, 1] / 255.0 + 1e-3)[0]
        hue = hue / (hue.sum() + 1e-6)
        sat = np.histogram(pix[:, 1], bins=3, range=(0, 256))[0] / len(pix)
        val = np.histogram(pix[:, 2], bins=3, range=(0, 256))[0] / len(pix)
        shape = np.array([np.log(area_frac * 1000 + 1) / 5, np.tanh(np.log(w / h)), cy])
        return np.concatenate([hue * 1.5, sat, val, shape * 0.7]).astype(np.float32)

    def _assign(self, d: np.ndarray) -> int:
        if self.centroids:
            dist = np.linalg.norm(np.stack(self.centroids) - d, axis=1)
            k = int(dist.argmin())
            if dist[k] < self.new_type_distance or len(self.centroids) >= self.n_types:
                self.counts[k] += 1
                lr = max(0.01, 1.0 / self.counts[k])
                self.centroids[k] += lr * (d - self.centroids[k])  # k-moyennes en ligne
                return k
        self.centroids.append(d.copy())
        self.counts.append(1)
        self.gain.append(0.0)
        self.danger.append(0.0)
        self.touches.append(0)
        return len(self.centroids) - 1

    def update(self, frame_bgr: np.ndarray) -> list[Element]:
        H, W = frame_bgr.shape[:2]
        s = self.work_width / W
        small = cv2.resize(frame_bgr, (self.work_width, max(1, int(H * s))), interpolation=cv2.INTER_AREA)
        f = cv2.GaussianBlur(small, (3, 3), 0)
        if self.frames and self.frames[0].shape != f.shape:
            self.frames.clear()
        self._t += 1
        if self._t % 2 == 0 or len(self.frames) < 5:
            self.frames.append(f)
        if len(self.frames) < 5:
            return []
        if self.bg is None or self._t % 3 == 0:
            self.bg = np.median(np.stack(self.frames), axis=0).astype(np.int16)
        diff = np.abs(f.astype(np.int16) - self.bg).max(axis=2)
        mask = (diff > self.diff_threshold).astype(np.uint8) * 255
        mask[: int(self.ignore_top * mask.shape[0])] = 0
        if self.roi is not None:
            if self._roi_mask is None or self._roi_mask.shape != mask.shape:
                self._roi_mask = np.zeros_like(mask)
                pts = (self.roi * [mask.shape[1], mask.shape[0]]).astype(np.int32)
                cv2.fillPoly(self._roi_mask, [pts], 255)
            mask &= self._roi_mask
        if mask.mean() > 0.6 * 255:  # toute l'image a changé (secousse, menu...)
            self.frames.clear()
            self.bg = None
            return []
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        sh, sw = mask.shape
        out = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            area_frac = area / (sh * sw)
            if area_frac < self.min_area or area_frac > 0.4:
                continue
            d = self._descriptor(hsv[y:y + h, x:x + w], (labels[y:y + h, x:x + w] == i), w, h,
                                 (y + h / 2) / sh, area_frac)
            k = self._assign(d)
            out.append(Element((int(x / s), int(y / s), int(w / s), int(h / s)), k))
        return out

    # ------------------------------------------------------------ ce que vaut chaque type
    def value(self, k: int) -> float:
        return self.gain[k] + self.danger[k]

    def category(self, k: int) -> str:
        """Ce que l'IA a déduit de ce type, en le comparant à la moyenne des autres."""
        if self.touches[k] < self.min_touches:
            return "?"
        if self.danger[k] < self.avg_danger * self.obstacle_factor - 0.05:
            return "OBSTACLE"   # plus souvent suivi d'un choc / d'une mort que la moyenne
        if self.gain[k] > self.avg_gain * self.bonus_factor + 0.05:
            return "BONUS"      # plus souvent suivi de pièces / d'étoiles que la moyenne
        return "neutre"

    def learn_values(self, elements: list[Element], frame_shape, gain: float, danger: float,
                     episode_end: bool = False, player_zone=None):
        """Relie chaque contact personnage/élément aux récompenses qui suivent.

        Le gain est compté sur ``bonus_horizon`` pas (un aimant rapporte des
        pièces APRÈS avoir été pris), le danger sur ``danger_horizon`` pas seulement.
        """
        H, W = frame_shape[:2]
        x0, y0, x1, y1 = player_zone or self.player_zone
        px0, py0, px1, py1 = x0 * W, y0 * H, x1 * W, y1 * H
        here = set()
        for e in elements:
            x, y, w, h = e.box
            if x < px1 and x + w > px0 and y < py1 and y + h > py0:
                here.add(e.type_id)
        for k in here - self._prev_here:  # nouveau contact
            self._open.append([k, 0, 0.0, 0.0])
        self._prev_here = here
        self._open = self._open[-300:]

        still_open = []
        for rec in self._open:
            k, age = rec[0], rec[1]
            rec[2] += gain * (0.97 ** age)
            if age < self.danger_horizon:
                rec[3] += danger
            rec[1] += 1
            if rec[1] >= self.bonus_horizon or episode_end:
                self._close(rec)
            else:
                still_open.append(rec)
        self._open = still_open
        if episode_end:
            self._prev_here = set()

    def _close(self, rec):
        k, _, g, d = rec
        self.touches[k] += 1
        lr = max(0.02, 1.0 / self.touches[k])
        self.gain[k] += lr * (g - self.gain[k])
        self.danger[k] += lr * (d - self.danger[k])
        self.avg_gain += 0.01 * (g - self.avg_gain)
        self.avg_danger += 0.01 * (d - self.avg_danger)


def agent_attention(model, obs) -> np.ndarray | None:
    """Carte (H, W) dans [0, 1] des pixels qui influencent le plus l'action choisie."""
    try:
        policy = model.policy
        x, _ = policy.obs_to_tensor(obs)
        x = x[:1].float().detach().requires_grad_(True)
        dist = policy.get_distribution(x)
        logits = dist.distribution.logits
        logits[0, logits[0].argmax()].backward()
        g = x.grad[0].abs().sum(dim=0).cpu().numpy()  # (H, W) après mise en canaux d'abord
        g = cv2.GaussianBlur(g, (0, 0), 2)
        return g / (g.max() + 1e-8)
    except Exception:
        return None


class LiveOverlay:
    """Fenêtre temps réel : image du jeu + cadres des éléments + attention de l'agent."""

    def __init__(self, cfg: dict | None = None, window_pos=None, video_path: str | None = None,
                 show: bool = True, crop_y=(0.0, 1.0)):
        cfg = cfg or {}
        self.discovery = ElementDiscovery(**cfg.get("discovery", {}))
        self.display_height = cfg.get("display_height", 900)
        self.show_attention = cfg.get("show_attention", True)
        self.every = max(1, cfg.get("every_n_steps", 1))
        self.window_pos, self.show, self.crop_y = window_pos, show, crop_y
        self.video_path, self.writer = video_path, None
        self.step = 0
        self.title = "IA Subway Surfers - elements"

    def update(self, frame_bgr, info: dict | None = None, model=None, obs=None, action=None):
        self.step += 1
        if frame_bgr is None:
            return
        elements = self.discovery.update(frame_bgr)
        info = info or {}
        parts = info.get("reward_parts", {})
        gain = parts.get("coins", 0.0) + parts.get("stars", 0.0)
        danger = parts.get("stumble", 0.0) + parts.get("death", 0.0)
        self.discovery.learn_values(elements, frame_bgr.shape, gain, danger,
                                    episode_end="episode_totals" in info,
                                    player_zone=info.get("player_zone"))
        if self.step % self.every:
            return

        img = frame_bgr.copy()
        H, W = img.shape[:2]
        att = agent_attention(model, obs) if (self.show_attention and model is not None and obs is not None) else None
        if att is not None:
            y0, y1 = int(self.crop_y[0] * H), int(self.crop_y[1] * H)
            att_full = np.zeros((H, W), np.float32)
            att_full[y0:y1] = cv2.resize(att, (W, y1 - y0))
            heat = cv2.applyColorMap((att_full * 255).astype(np.uint8), cv2.COLORMAP_JET)
            a = (np.clip(att_full * 1.5, 0, 1) * 0.45)[..., None]
            img = (img * (1 - a) + heat * a).astype(np.uint8)
            for e in elements:
                x, y, w, h = e.box
                e.attention = float(att_full[y:y + h, x:x + w].mean()) if w and h else 0.0

        scale = self.display_height / H
        img = cv2.resize(img, (int(W * scale), self.display_height))
        for e in elements:
            x, y, w, h = (int(v * scale) for v in e.box)
            c = PALETTE[e.type_id % len(PALETTE)]
            cat = self.discovery.category(e.type_id)
            if cat == "OBSTACLE":  # double cadre rouge autour des obstacles
                cv2.rectangle(img, (x - 3, y - 3), (x + w + 3, y + h + 3), (0, 0, 255), 2)
            elif cat == "BONUS":  # double cadre vert autour des bonus
                cv2.rectangle(img, (x - 3, y - 3), (x + w + 3, y + h + 3), (0, 220, 0), 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), c, 2)
            label = f"T{e.type_id} {cat if cat != '?' else ''} {self.discovery.value(e.type_id):+.1f}"
            if att is not None:
                label += f" att {e.attention * 100:.0f}%"
            (tw, th_), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (x, max(0, y - th_ - 4)), (x + tw + 4, y), c, -1)
            cv2.putText(img, label, (x + 2, max(th_, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 0), 1, cv2.LINE_AA)

        img = np.hstack([img, self._panel(img.shape[0], info, action)])
        if self.video_path:
            if self.writer is None:
                self.writer = cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10,
                                              (img.shape[1], img.shape[0]))
            self.writer.write(img)
        if self.show:
            cv2.imshow(self.title, img)
            if self.window_pos and self.step <= self.every:
                cv2.moveWindow(self.title, *self.window_pos)
            cv2.waitKey(1)

    def _panel(self, h, info, action):
        p = np.full((h, 260, 3), 25, np.uint8)
        put = lambda t, y, c=(230, 230, 230), s=0.5: cv2.putText(  # noqa: E731
            p, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, s, c, 1, cv2.LINE_AA)
        put("Elements decouverts", 25, s=0.6)
        put("(apres contact avec le personnage)", 45, (160, 160, 160), 0.4)
        d, y = self.discovery, 75
        order = sorted(range(len(d.centroids)), key=lambda k: -d.counts[k])
        for k in order:
            c = PALETTE[k % len(PALETTE)]
            cv2.rectangle(p, (10, y - 11), (24, y + 3), c, -1)
            put(f"T{k} {d.category(k):<8} gain {d.gain[k]:+.2f} danger {d.danger[k]:+.2f}", y + 1, s=0.4)
            y += 22
        y += 15
        if action is not None:
            put(f"action : {ACTION_NAMES[int(action)]}", y)
            y += 22
        for key, name in (("score", "score"), ("coins", "pieces"), ("stars", "etoiles")):
            if key in info:
                put(f"{name} : {info[key]}", y)
                y += 22
        if self.show_attention:
            put("couleurs chaudes = ou l'IA regarde", h - 15, (160, 160, 160), 0.4)
        return p

    def close(self):
        if self.writer is not None:
            self.writer.release()
        if self.show:
            cv2.destroyAllWindows()
