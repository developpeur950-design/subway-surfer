"""Lecture des chiffres du HUD en apprenant la police du jeu.

Tesseract lit mal la police stylisée de Subway Surfers City (le « 1 » ressemble à un « l »,
la virgule de « x8,5 » disparaît...). Ici, on apprend la forme exacte de chaque caractère à
partir de quelques captures où l'utilisateur tape ce qui est affiché (``calibrate.py digits``),
puis on lit en découpant les caractères et en les comparant aux formes apprises.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import cv2
import numpy as np

SIZE = 28
NAMES = {",": "virgule", ".": "point", "x": "x"}
# le score et les pièces utilisent la même police ; le multiplicateur une autre
GROUPS = {"score": "hud", "coins": "hud", "multiplier": "mult"}
MODES = ("enclosed", "white", "otsu")


@dataclass
class Glyph:
    x: int
    shape: np.ndarray  # forme normalisée SIZE x SIZE
    rel_h: float       # hauteur / hauteur de la ligne (virgule ~0.4, « x » ~0.7, chiffre ~1)
    rel_cy: float      # centre vertical dans la ligne (virgule en bas ~0.8)


def _mask(roi: np.ndarray, mode: str) -> np.ndarray:
    v = roi.max(axis=2)
    if mode == "white":
        m = roi.min(axis=2) > 170
    elif mode == "green":
        b, g, r = roi[..., 0].astype(int), roi[..., 1].astype(int), roi[..., 2].astype(int)
        m = (g > 140) & (g > r + 25) & (g > b + 25)
    else:
        _, t = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        m = t > 0
        if mode == "enclosed":
            # texte clair entouré d'un contour sombre, quelle que soit sa couleur : on garde
            # les zones claires qui ne touchent aucun bord (le décor clair, lui, les touche)
            n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=4)
            H, W = m.shape
            keep = np.zeros(n, bool)
            for i in range(1, n):
                x, y, w, h, _ = st[i]
                keep[i] = x > 0 and y > 0 and x + w < W and y + h < H
            m = keep[lab]
    m = m.astype(np.uint8) * 255
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def _normalize(g: np.ndarray) -> np.ndarray:
    h, w = g.shape
    s = (SIZE - 2) / max(h, w)
    g = cv2.resize(g, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((SIZE, SIZE), np.float32)
    y0, x0 = (SIZE - g.shape[0]) // 2, (SIZE - g.shape[1]) // 2
    canvas[y0:y0 + g.shape[0], x0:x0 + g.shape[1]] = g / 255.0
    return canvas


def segment(roi: np.ndarray, mode: str) -> list[Glyph]:
    m = _mask(roi, mode)
    H, W = m.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    # ce qui touche un bord de la zone = décor ou caractère coupé
    comps = [(x, y, w, h, i) for i, (x, y, w, h, area) in enumerate(stats) if i > 0
             and area >= 4 and y > 0 and y + h < H and x > 0 and x + w < W]
    # trous des chiffres (intérieur d'un 8, d'un 0...) : contenus dans un autre caractère
    comps = [c for c in comps if not any(o is not c and o[0] <= c[0] and o[1] <= c[1]
                                         and c[0] + c[2] <= o[0] + o[2] and c[1] + c[3] <= o[1] + o[3]
                                         for o in comps)]
    if not comps:
        return []
    big = max(c[3] for c in comps)
    # trop large pour un caractère (bord d'un cadre du HUD...) ou minuscule
    comps = sorted(c for c in comps if c[3] >= 0.2 * big and c[2] <= 1.2 * big)
    if not comps:
        return []
    big = max(c[3] for c in comps)
    # morceaux isolés aux extrémités, loin du texte
    while len(comps) > 1 and comps[1][0] - (comps[0][0] + comps[0][2]) > 0.7 * big:
        comps.pop(0)
    while len(comps) > 1 and comps[-1][0] - (comps[-2][0] + comps[-2][2]) > 0.7 * big:
        comps.pop()
    big = max(c[3] for c in comps)
    top = min(c[1] for c in comps if c[3] >= 0.6 * big)
    line_h = max(1, max(c[1] + c[3] for c in comps) - top)
    out = []
    for x, y, w, h, i in sorted(comps):
        g = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        out.append(Glyph(x, _normalize(g), h / line_h, (y + h / 2 - top) / line_h))
    return out


def _sim(a: np.ndarray, b: np.ndarray) -> float:
    """Ressemblance entre deux formes, tolérante à un petit décalage et au flou de taille."""
    a = cv2.GaussianBlur((a > 0.5).astype(np.float32), (3, 3), 0)
    b = cv2.GaussianBlur((b > 0.5).astype(np.float32), (3, 3), 0)
    pad = np.pad(a, 2)
    res = cv2.matchTemplate(pad, b, cv2.TM_CCOEFF_NORMED)
    return float(res.max())


class GlyphReader:
    def __init__(self, folder: str, group: str):
        self.folder = os.path.join(folder, group)
        self.samples: dict[str, list[tuple[np.ndarray, float, float]]] = {}
        self.mode, self.feats, self.mode_ok = "enclosed", {}, {}
        meta = os.path.join(self.folder, "meta.json")
        if os.path.exists(meta):
            with open(meta, encoding="utf-8") as f:
                d = json.load(f)
            self.mode, self.feats = d.get("mode", "enclosed"), d.get("feats", {})
            self.mode_ok = d.get("mode_ok", {})
        back = {v: k for k, v in NAMES.items()}
        for fn, (rh, rc) in self.feats.items():
            img = cv2.imread(os.path.join(self.folder, fn), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                ch = back.get(fn.rsplit("_", 1)[0], fn.rsplit("_", 1)[0])
                self.samples.setdefault(ch, []).append((img.astype(np.float32) / 255.0, rh, rc))

    @property
    def ready(self) -> bool:
        return not self.missing()

    def missing(self) -> list[str]:
        return [d for d in "0123456789" if d not in self.samples]

    # ------------------------------------------------------------ apprentissage
    def learn(self, roi: np.ndarray, text: str) -> bool:
        """Apprend à partir d'une capture dont on connaît le texte (ex : « x8,5 », « 42 471 »)."""
        chars = list(text.lower().replace(" ", ""))
        # découpages qui retrouvent le bon nombre de caractères ; on garde le mode qui
        # marche sur le plus d'exemples
        ok = {m: segment(roi, m) for m in MODES}
        ok = {m: g for m, g in ok.items() if len(g) == len(chars)}
        if not ok:
            return False
        for m in ok:
            self.mode_ok[m] = self.mode_ok.get(m, 0) + 1
        best = max(MODES, key=lambda m: (self.mode_ok.get(m, 0), -MODES.index(m)))
        self.mode = best
        glyphs = ok.get(best) or next(iter(ok.values()))
        os.makedirs(self.folder, exist_ok=True)
        for g, ch in zip(glyphs, chars):
            lst = self.samples.setdefault(ch, [])
            if len(lst) >= 6:
                continue
            lst.append((g.shape, g.rel_h, g.rel_cy))
            fn = f"{NAMES.get(ch, ch)}_{len(lst)}.png"
            cv2.imwrite(os.path.join(self.folder, fn), (g.shape * 255).astype(np.uint8))
            self.feats[fn] = [g.rel_h, g.rel_cy]
        with open(os.path.join(self.folder, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"mode": self.mode, "mode_ok": self.mode_ok, "feats": self.feats}, f, indent=1)
        return True

    # ------------------------------------------------------------ lecture
    def read(self, roi: np.ndarray, min_score: float = 0.5) -> str | None:
        """Lit le texte de la zone, ou None si un caractère n'est pas reconnu avec confiance.

        Le découpage appris est essayé en premier ; si l'apparence du HUD change (cadre qui
        s'illumine pendant un bonus...), les autres découpages sont essayés aussi.
        """
        if not self.samples:
            return None
        order = [self.mode] + sorted((m for m in MODES if m != self.mode),
                                     key=lambda m: -self.mode_ok.get(m, 0))
        for mode in order:
            text = self._read_mode(roi, mode, min_score)
            if text:
                return text
        return None

    def _read_mode(self, roi, mode, min_score):
        glyphs = segment(roi, mode)
        if not glyphs:
            return None
        text = []
        for g in glyphs:
            if g.rel_h < 0.55 and g.rel_cy > 0.6 and not any(c in self.samples for c in ",."):
                text.append(",")  # aucune virgule apprise : petit et en bas = virgule
                continue
            best, best_ch = -9.0, None
            for ch, lst in self.samples.items():
                for shape, rh, rc in lst:
                    s = _sim(g.shape, shape) - 1.5 * (abs(g.rel_h - rh) + abs(g.rel_cy - rc))
                    if s > best:
                        best, best_ch = s, ch
            if best < min_score:
                return None  # caractère inconnu : on préfère ne rien lire
            text.append(best_ch)
        return "".join(text)
