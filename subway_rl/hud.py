"""Lecture du HUD du vrai jeu à partir de captures d'écran.

- Score, pièces et multiplicateur : OCR (Tesseract) sur des zones de l'écran
  définies dans config.yaml (voir ``calibrate.py``).
- Mort et trébuchement (choc contre un mur) : détection par modèles d'images
  (template matching) capturés une fois avec ``calibrate.py``, plus une
  heuristique de secours pour la mort (image figée).

Ceci sert UNIQUEMENT à calculer les récompenses. L'agent, lui, ne voit que les
pixels et apprend seul à reconnaître trains, barrières, pièces, étoiles...
"""
from __future__ import annotations

import os
import re
import shutil

import cv2
import numpy as np

from .rewards import HudState

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None

if pytesseract is not None and shutil.which("tesseract") is None:
    # Windows : Tesseract n'est souvent pas dans le PATH, on le cherche à sa place habituelle
    for exe in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
        if os.path.exists(exe):
            pytesseract.pytesseract.tesseract_cmd = exe
            break


def crop(img: np.ndarray, region) -> np.ndarray:
    """region = [x, y, largeur, hauteur] relative à la fenêtre du jeu."""
    x, y, w, h = region
    return img[y:y + h, x:x + w]


class NumberReader:
    """Lit un nombre affiché à l'écran, avec un filtre contre les erreurs d'OCR."""

    def __init__(self, region, monotonic: bool = True, max_jump: float | None = None,
                 invert: bool = False, scale: float = 3.0, decimal: bool = False):
        if pytesseract is None:
            raise ImportError("pytesseract n'est pas installé (pip install pytesseract + binaire Tesseract)")
        self.region, self.monotonic, self.max_jump = region, monotonic, max_jump
        self.invert, self.scale, self.decimal = invert, scale, decimal
        self.value, self._fresh = 0, True

    def reset(self, value: int = 0):
        self.value = value
        self._fresh = True  # la première lecture d'une partie est acceptée telle quelle

    def raw_read(self, img: np.ndarray) -> float | None:
        roi = crop(img, self.region)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
        # Le HUD de Subway Surfers = chiffres blancs avec contour sombre
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if not self.invert:
            bw = 255 - bw  # Tesseract préfère du texte noir sur fond blanc
        whitelist = "0123456789xX" + (",." if self.decimal else "")
        text = pytesseract.image_to_string(
            bw, config=f"--psm 7 -c tessedit_char_whitelist={whitelist}")
        if self.decimal:  # ex : "x8,5" -> 8.5
            m = re.search(r"\d+(?:[.,]\d+)?", text.replace(" ", ""))
            return float(m.group().replace(",", ".")) if m else None
        digits = re.sub(r"\D", "", text)  # ex : "151 877" -> 151877
        return int(digits) if digits else None

    def read(self, img: np.ndarray) -> float:
        v = self.raw_read(img)
        if v is None:
            return self.value
        if self._fresh:
            self.value, self._fresh = v, False
            return v
        if self.monotonic and v < self.value:
            return self.value
        if self.max_jump is not None and self.value and v - self.value > self.max_jump:
            return self.value
        self.value = v
        return v


class TemplateDetector:
    """Détecte une image de référence (ex : écran "Game Over", flash de choc)."""

    def __init__(self, path: str, region=None, threshold: float = 0.8):
        tpl = cv2.imread(path, cv2.IMREAD_COLOR)
        if tpl is None:
            raise FileNotFoundError(path)
        self.tpl, self.region, self.threshold = tpl, region, threshold

    def score(self, img: np.ndarray) -> float:
        area = crop(img, self.region) if self.region else img
        if area.shape[0] < self.tpl.shape[0] or area.shape[1] < self.tpl.shape[1]:
            return 0.0
        res = cv2.matchTemplate(area, self.tpl, cv2.TM_CCOEFF_NORMED)
        return float(res.max())

    def __call__(self, img: np.ndarray) -> bool:
        return self.score(img) >= self.threshold


class HudReader:
    def __init__(self, cfg: dict):
        regions = cfg.get("regions", {})
        self.score = NumberReader(regions["score"], max_jump=cfg.get("max_score_jump", 5000))
        self.coins = NumberReader(regions["coins"], max_jump=50)
        # Le multiplicateur est lu pour information (il peut changer pour d'autres
        # raisons que les étoiles) ; les étoiles sont détectées visuellement, voir read().
        self.mult = (NumberReader(regions["multiplier"], monotonic=False, max_jump=100, decimal=True)
                     if regions.get("multiplier") else None)

        def detectors(key):
            out = []
            for t in cfg.get("templates", {}).get(key, []):
                if os.path.exists(t["path"]):
                    out.append(TemplateDetector(t["path"], t.get("region"), t.get("threshold", 0.8)))
                else:
                    print(f"[hud] modèle introuvable, ignoré : {t['path']}")
            return out

        self.death_detectors = detectors("death")
        self.stumble_detectors = detectors("stumble")
        self.star_detectors = detectors("star")
        self.frozen_frames_for_death = cfg.get("frozen_frames_for_death", 15)
        self.stumble_cooldown = cfg.get("stumble_cooldown_steps", 10)
        self.star_cooldown = cfg.get("star_cooldown_steps", 3)
        if not self.star_detectors:
            print("[hud] aucune image 'star' : les étoiles ne seront pas récompensées "
                  "(voir calibrate.py record / template star)")
        self.reset()

    def reset(self):
        self.score.reset(0)
        self.coins.reset(0)
        if self.mult:
            self.mult.reset(1)
        self.multiplier, self.stars = None, 0  # None = valeur de base pas encore lue
        self._star_seen = False
        self._prev_small, self._frozen = None, 0
        self._stumble_cd, self._star_cd = 0, 0

    def _is_frozen(self, img) -> bool:
        small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (64, 64)).astype(np.int16)
        frozen = self._prev_small is not None and np.abs(small - self._prev_small).mean() < 1.0
        self._prev_small = small
        self._frozen = self._frozen + 1 if frozen else 0
        return self._frozen >= self.frozen_frames_for_death

    def read(self, img: np.ndarray) -> HudState:
        dead = any(d(img) for d in self.death_detectors) or self._is_frozen(img)

        stumbled = False
        self._stumble_cd = max(0, self._stumble_cd - 1)
        if self._stumble_cd == 0 and any(d(img) for d in self.stumble_detectors):
            stumbled, self._stumble_cd = True, self.stumble_cooldown

        if self.mult:
            self.multiplier = max(1, self.mult.read(img))

        # Étoile ramassée : l'effet visuel de ramassage (image de référence "star")
        # apparaît. On compte le front montant (absent -> présent) pour ne pas compter
        # plusieurs fois la même animation qui dure plusieurs images.
        self._star_cd = max(0, self._star_cd - 1)
        seen = any(d(img) for d in self.star_detectors)
        if seen and not self._star_seen and self._star_cd == 0:
            self.stars += 1
            self._star_cd = self.star_cooldown
        self._star_seen = seen

        return HudState(score=self.score.read(img), coins=self.coins.read(img),
                        multiplier=self.multiplier or 1, stars=self.stars,
                        dead=dead, stumbled=stumbled)
