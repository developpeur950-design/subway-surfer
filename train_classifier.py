"""Entraîne un détecteur « oui / non » sur les images étiquetées avec « calibrate.py label <nom> ».

    python train_classifier.py garde      # le garde est-il visible ? (sert à détecter les chocs)

Conseil : au moins 40 images « oui » et autant de « non », issues de plusieurs parties
(chocs dans différents décors, de jour, dans les tunnels, avec effets...).
"""
from __future__ import annotations

import argparse

import numpy as np

from subway_rl import event_classifier as ec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="garde")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--zone", type=float, nargs=4, help="zone regardée x0 y0 x1 y1 (fractions)")
    args = ap.parse_args()
    imgs, ys = ec.load_labels(args.name)
    n_yes = int(np.sum(ys))
    if n_yes < 10 or len(ys) - n_yes < 10:
        raise SystemExit(f"{n_yes} « oui » et {len(ys) - n_yes} « non » : il en faut au moins 10 de chaque "
                         f"(40+ conseillé). Lancez python calibrate.py label {args.name}")
    zone = tuple(args.zone) if args.zone else ec.DEFAULT_ZONES.get(args.name, (0, 0, 1, 1))
    print(f"Entraînement sur {len(ys)} images ({n_yes} oui)...")
    net = ec.train(imgs, ys, zone, steps=args.steps)
    out = f"models/{args.name}_classifier.pt"
    ec.save(net, zone, out)
    c = ec.EventClassifier(out)
    acc = np.mean([(c.prob(i) >= 0.5) == bool(y) for i, y in zip(imgs, ys)])
    print(f"Modèle enregistré : {out} (juste sur {acc:.0%} des images apprises)")


if __name__ == "__main__":
    main()
