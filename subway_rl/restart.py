"""Relance automatique des parties, en reconnaissant les boutons du jeu à l'écran.

Après une mort, Subway Surfers City enchaîne plusieurs écrans dont la durée varie
(« Continuer la course ? », scores, chargement, menu, parfois une publicité ou une
récompense). Plutôt que des clics à l'aveugle, on regarde l'écran en boucle : dès qu'un
bouton connu apparaît, on clique dessus, jusqu'à ce que le bouton pause (en jeu) revienne.

Boutons connus = images de référence capturées avec « calibrate.py template <nom> » :
  terminer   le texte « APPUIE POUR TERMINER LA COURSE » (surtout PAS le bouton qui
             dépense des clés pour continuer)
  continuer  le bouton orange CONTINUER de l'écran des scores
  course     le bouton COURSE du menu
  fermer     la croix d'une fenêtre surprise (publicité, récompense...)
L'ordre et la liste se règlent dans config.yaml (game.restart_buttons).
"""
from __future__ import annotations

import time

from .hud import TemplateDetector


class Restarter:
    def __init__(self, cfg: dict, controller, capture, hud):
        g = cfg
        self.controller, self.capture, self.hud = controller, capture, hud
        self.timeout = g.get("restart_timeout", 60)
        self.click_wait = g.get("restart_click_wait", 1.5)
        self.fallback = g.get("restart_sequence")
        self.buttons = []
        for name in g.get("restart_buttons", ["terminer", "continuer", "course", "fermer"]):
            for t in g.get("templates", {}).get(name) or []:
                try:
                    self.buttons.append((name, TemplateDetector(t["path"], t.get("region"),
                                                                t.get("threshold", 0.8))))
                except FileNotFoundError:
                    print(f"[relance] image introuvable, ignorée : {t['path']}")

    @property
    def smart(self) -> bool:
        return bool(self.buttons) and bool(self.hud.ingame_detectors)

    def restart(self) -> bool:
        """Revient en jeu. Renvoie True si une partie est en cours à la fin."""
        if not self.smart:  # ancienne méthode : clics à des positions fixes
            self.controller.run_sequence(self.fallback)
            return True
        t0 = last_action = time.time()
        seen_ingame = 0
        while time.time() - t0 < self.timeout:
            img = self.capture.grab()
            if self.hud.in_game(img):
                seen_ingame += 1
                if seen_ingame >= 2:  # bouton pause vu 2 fois d'affilée : on joue
                    return True
                time.sleep(0.2)
                continue
            seen_ingame = 0
            for name, det in self.buttons:
                pos = det.locate(img)
                if pos is not None:
                    print(f"[relance] clic sur « {name} »")
                    self.controller.click(*pos)
                    time.sleep(self.click_wait)
                    last_action = time.time()
                    break
            else:
                # aucun bouton connu depuis un moment : clics de secours (config restart_sequence)
                if time.time() - last_action > 10 and self.fallback:
                    print("[relance] aucun bouton reconnu : clics de secours")
                    self.controller.run_sequence(self.fallback)
                    last_action = time.time()
                time.sleep(0.4)
        print("[relance] impossible de relancer une partie : capturez le bouton bloquant avec "
              "« python calibrate.py template <nom> » et ajoutez-le à game.restart_buttons")
        return False
