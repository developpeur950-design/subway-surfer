"""Outils pour configurer le vrai jeu.

    python calibrate.py screenshot                 # capture la zone "capture" -> calib/screen.png
    python calibrate.py region score               # sélection à la souris de la zone du score
    python calibrate.py region coins
    python calibrate.py region multiplier
    python calibrate.py template death --image calib/screen.png    # découpe une image de référence
    python calibrate.py test                       # affiche en direct ce que lit le HUD
    python calibrate.py keys                       # teste les commandes (gauche, droite, saut, roulade)

Astuce : pour trouver la zone "capture", lancez `python calibrate.py screenshot --full`
puis `python calibrate.py region capture --image calib/full.png`.
"""
from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np
import yaml

import subway_rl.controls  # noqa: F401  (active les coordonnées en pixels réels sous Windows)

CONFIG = "config.yaml"


def load(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save(cfg, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def grab(cfg, full=False):
    import mss
    with mss.mss() as sct:
        mon = sct.monitors[1] if full else cfg["game"]["capture"]
        return np.asarray(sct.grab(mon))[:, :, :3].copy()


def select(img, title):
    print("Dessinez un rectangle à la souris puis appuyez sur ENTRÉE (ÉCHAP pour annuler).")
    r = cv2.selectROI(title, img, showCrosshair=True)
    cv2.destroyAllWindows()
    if r[2] == 0 or r[3] == 0:
        raise SystemExit("Annulé.")
    return [int(v) for v in r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["screenshot", "region", "template", "test", "keys"])
    ap.add_argument("name", nargs="?")
    ap.add_argument("--image")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--config", default=CONFIG)
    args = ap.parse_args()
    cfg = load(args.config)
    os.makedirs("calib", exist_ok=True)

    if args.cmd == "screenshot":
        out = "calib/full.png" if args.full else "calib/screen.png"
        print("Capture dans 3 s...")
        time.sleep(3)
        cv2.imwrite(out, grab(cfg, args.full))
        print(f"-> {out}")

    elif args.cmd == "region":
        img = cv2.imread(args.image or "calib/screen.png")
        x, y, w, h = select(img, f"Zone : {args.name}")
        if args.name == "capture":
            cfg["game"]["capture"] = {"left": x, "top": y, "width": w, "height": h}
        else:
            cfg["game"].setdefault("regions", {})[args.name] = [x, y, w, h]
        save(cfg, args.config)
        print(f"{args.name} = {[x, y, w, h]} enregistré dans {args.config}")

    elif args.cmd == "template":
        img = cv2.imread(args.image or "calib/screen.png")
        x, y, w, h = select(img, f"Modèle : {args.name}")
        os.makedirs("templates", exist_ok=True)
        path = f"templates/{args.name}.png"
        cv2.imwrite(path, img[y:y + h, x:x + w])
        lst = cfg["game"].setdefault("templates", {}).setdefault(args.name, [])
        if not any(t["path"] == path for t in lst):
            lst.append({"path": path, "threshold": 0.8})
        save(cfg, args.config)
        print(f"Modèle enregistré : {path}")

    elif args.cmd == "test":
        from subway_rl.hud import HudReader
        hud = HudReader(cfg["game"])
        print("Lecture du HUD en direct (Ctrl+C pour arrêter)")
        while True:
            img = grab(cfg)
            s = hud.read(img)
            extra = " ".join(f"{d.score(img):.2f}" for d in hud.death_detectors)
            print(f"score={s.score:>8} pièces={s.coins:>5} mult=x{s.multiplier:<3} étoiles={s.stars} "
                  f"mort={s.dead} choc={s.stumbled}  [similarité mort: {extra}]")
            time.sleep(0.2)

    elif args.cmd == "keys":
        from subway_rl.controls import ACTION_NAMES, make_controller
        ctrl = make_controller(cfg["game"])
        print("Cliquez dans la fenêtre du jeu, test dans 3 s...")
        time.sleep(3)
        for a in [1, 2, 2, 1, 3, 4]:
            print(ACTION_NAMES[a])
            ctrl.act(a)
            time.sleep(0.7)


if __name__ == "__main__":
    main()
