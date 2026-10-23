"""Entraîne le détecteur du personnage sur les images encadrées avec « calibrate.py player ».

    python train_player.py             # ~1500 étapes : quelques minutes avec une carte NVIDIA,
                                       # 15-30 minutes sur le processeur
"""
from __future__ import annotations

import argparse

from subway_rl import player_detector as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--out", default="models/player_detector.pt")
    args = ap.parse_args()
    imgs, boxes = pd.load_labels()
    if len(imgs) < 15:
        raise SystemExit(f"Seulement {len(imgs)} images encadrées : il en faut au moins 15 "
                         "(30 ou plus conseillé). Lancez python calibrate.py player")
    print(f"Entraînement sur {len(imgs)} images...")
    net = pd.train(imgs, boxes, steps=args.steps)
    pd.save(net, args.out)
    det = pd.PlayerDetector(args.out, min_conf=0.0)
    ok = 0
    for img, (x0, y0, x1, y1) in zip(imgs, boxes):
        b = det.find(img)
        cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
        ok += x0 <= cx <= x1 and y0 <= cy <= y1
    print(f"Modèle enregistré : {args.out} (retrouve le personnage sur {ok}/{len(imgs)} images apprises)")


if __name__ == "__main__":
    main()
