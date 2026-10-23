"""Récupère vos réglages de calibration d'un ancien config.yaml vers config.local.yaml.

À utiliser une seule fois, si vous aviez calibré avant que les réglages personnels soient
séparés (config.local.yaml) :

    copy config.yaml config.ancien.yaml
    git checkout -- config.yaml
    git pull origin claude/youthful-albattani-m7ol59
    python migrer_config.py config.ancien.yaml
"""
from __future__ import annotations

import sys

import yaml

from subway_rl.config import deep_merge, load_local, save_local

# réglages propres à VOTRE écran et à VOTRE partie (le reste vient des valeurs par défaut)
GAME_KEYS = ["controller", "capture", "regions", "templates", "restart_sequence",
             "pause_sequence", "resume_sequence", "mouse_swipe", "adb"]


def main():
    old_path = sys.argv[1] if len(sys.argv) > 1 else "config.ancien.yaml"
    with open(old_path, encoding="utf-8") as f:
        old = yaml.safe_load(f) or {}
    game = old.get("game", {})
    mine = {"game": {k: game[k] for k in GAME_KEYS if k in game}}
    home = old.get("overlay", {}).get("player_tracker", {}).get("home")
    if home:
        mine["overlay"] = {"player_tracker": {"home": home}}
    out = save_local(deep_merge(load_local(), mine))
    print(f"Réglages récupérés dans {out} : " + ", ".join(mine["game"]))


if __name__ == "__main__":
    main()
