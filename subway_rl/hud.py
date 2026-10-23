"""Lecture du HUD du vrai jeu à partir de captures d'écran.

- Score, pièces et multiplicateur : lus avec la police du jeu apprise par
  ``calibrate.py digits`` (voir glyphs.py), sinon OCR Tesseract, sur des zones de
  l'écran définies dans config.yaml (voir ``calibrate.py``).
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
                 invert: bool = False, scale: float = 3.0, decimal: bool = False, glyphs=None,
                 confirm: int = 1, valid_range: tuple | None = None):
        # glyphs : police du jeu apprise avec « calibrate.py digits » (bien plus fiable)
        self.glyphs = glyphs
        if pytesseract is None and glyphs is None:
            raise ImportError("pytesseract n'est pas installé (pip install pytesseract + binaire Tesseract)")
        self.region, self.monotonic, self.max_jump = region, monotonic, max_jump
        self.invert, self.scale, self.decimal = invert, scale, decimal
        self.value, self._fresh = 0, True
        self.confirm = confirm          # nb de lectures identiques d'affilée avant d'accepter
        self.valid_range = valid_range  # valeurs possibles (ex : multiplicateur entre 1 et 60)
        self._candidate, self._count = None, 0
        self._rejected: list[float] = []
        self.last_raw, self.status = None, ""  # pour le diagnostic (calibrate.py test)

    def reset(self, value: int = 0):
        self.value = value
        self._fresh = True  # la première lecture d'une partie est acceptée telle quelle

    def _parse(self, text: str) -> float | None:
        if self.decimal:  # ex : "x8,5" -> 8.5
            m = re.search(r"\d+(?:[.,]\d+)?", text.replace(" ", ""))
            return float(m.group().replace(",", ".")) if m else None
        digits = re.sub(r"\D", "", text)  # ex : "151 877" -> 151877
        return int(digits) if digits else None

    def raw_read(self, img: np.ndarray) -> float | None:
        roi = crop(img, self.region)
        if self.glyphs is not None and self.glyphs.samples:
            text = self.glyphs.read(roi)
            return self._parse(text) if text else None
        if pytesseract is None:
            return None
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
        # Le HUD de Subway Surfers = chiffres blancs avec contour sombre
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if not self.invert:
            bw = 255 - bw  # Tesseract préfère du texte noir sur fond blanc
        whitelist = "0123456789xX" + (",." if self.decimal else "")
        text = pytesseract.image_to_string(
            bw, config=f"--psm 7 -c tessedit_char_whitelist={whitelist}")
        return self._parse(text)

    def read(self, img: np.ndarray) -> float:
        v = self.raw_read(img)
        self.last_raw = v
        if v is not None and self.valid_range and not (self.valid_range[0] <= v <= self.valid_range[1]):
            # virgule lue comme un chiffre (« 12,8 » -> 128) : on tente de la remettre
            if self.decimal and v == int(v) and self.valid_range[0] <= v / 10 <= self.valid_range[1]:
                v = v / 10
            else:
                self.status = "hors limites"
                return self.value
        if v is None:
            self.status = "illisible"
            return self.value
        # score et pièces ne perdent jamais de chiffres pendant une partie : une lecture plus
        # courte est une lecture partielle (animation, chiffre masqué), jamais une vraie valeur
        if (self.monotonic and not self._fresh and self.value >= 10
                and len(str(int(v))) < len(str(int(self.value)))):
            self.status = "incomplet"
            return self.value
        # lecture confirmée plusieurs fois d'affilée (évite les erreurs isolées)
        self._count = self._count + 1 if v == self._candidate else 1
        self._candidate = v
        if self._count < self.confirm:
            self.status = "à confirmer"
            return self.value
        if self._fresh:
            self.value, self._fresh, self._rejected = v, False, []
            self.status = "ok"
            return v
        bad = ((self.monotonic and v < self.value)
               or (self.max_jump is not None and self.value and v - self.value > self.max_jump))
        if bad:
            # plusieurs lectures cohérentes entre elles mais rejetées : c'est la valeur
            # mémorisée qui était fausse -> on se recale dessus
            self._rejected = (self._rejected + [v])[-3:]
            r = self._rejected
            steady = len(r) == 3 and r[0] <= r[1] <= r[2] and (
                self.max_jump is None or (r[1] - r[0] <= self.max_jump / 4 and r[2] - r[1] <= self.max_jump / 4))
            if steady:
                self.value, self._rejected, self.status = v, [], "recalé"
                return v
            self.status = "rejeté"
            return self.value
        self._rejected = []
        self.value, self.status = v, "ok"
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
        from .glyphs import GROUPS, GlyphReader
        folder = cfg.get("digits_folder", "templates/digits")
        g = {grp: GlyphReader(folder, grp) for grp in set(GROUPS.values())}
        g = {k: (v if v.samples else None) for k, v in g.items()}
        self.score = NumberReader(regions["score"], max_jump=cfg.get("max_score_jump", 5000),
                                  glyphs=g[GROUPS["score"]])
        self.coins = NumberReader(regions["coins"], max_jump=50, glyphs=g[GROUPS["coins"]])
        # Le multiplicateur est lu pour information (il peut changer pour d'autres
        # raisons que les étoiles) ; les étoiles sont détectées visuellement, voir read().
        self.mult = (NumberReader(regions["multiplier"], monotonic=False, max_jump=100, decimal=True,
                                  glyphs=g[GROUPS["multiplier"]], confirm=2,
                                  valid_range=tuple(cfg.get("multiplier_range", [1, 60])))
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
