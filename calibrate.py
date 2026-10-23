"""Outils pour configurer le vrai jeu.

    python calibrate.py screenshot                 # capture la zone "capture" -> calib/screen.png
    python calibrate.py region score               # sélection à la souris de la zone du score
    python calibrate.py region coins
    python calibrate.py region multiplier
    python calibrate.py template death --image calib/screen.png    # découpe une image de référence
    python calibrate.py record --seconds 15        # enregistre des images -> calib/rec/ (pour les étoiles)
    python calibrate.py point                      # cliquez sur les boutons du jeu -> coordonnées
    python calibrate.py digits                     # apprend la police des chiffres du jeu
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
    with getattr(mss, "MSS", mss.mss)() as sct:
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
    ap.add_argument("cmd", choices=["screenshot", "record", "region", "template", "point", "digits", "test", "keys"])
    ap.add_argument("name", nargs="?")
    ap.add_argument("--image")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--margin", type=int, default=40,
                    help="template : marge (px) autour de la découpe où chercher l'image en jeu")
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

    elif args.cmd == "record":
        # Pour capturer un événement bref (ramassage d'une étoile, choc...) :
        # jouez pendant l'enregistrement puis choisissez la bonne image dans calib/rec/
        out = "calib/rec"
        os.makedirs(out, exist_ok=True)
        print(f"Enregistrement de {args.seconds:.0f} s dans 3 s... ramassez une étoile !")
        time.sleep(3)
        t0, i = time.time(), 0
        while time.time() - t0 < args.seconds:
            cv2.imwrite(f"{out}/{i:04d}.png", grab(cfg))
            i += 1
            time.sleep(0.05)
        print(f"{i} images dans {out}/ : trouvez celle où l'étoile est ramassée, puis\n"
              f"  python calibrate.py template star --image {out}/XXXX.png")

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
        lst = cfg["game"].setdefault("templates", {}).setdefault(args.name, [])
        # plusieurs images possibles par événement : star.png, star_2.png, ...
        path = f"templates/{args.name}.png" if not lst else f"templates/{args.name}_{len(lst) + 1}.png"
        cv2.imwrite(path, img[y:y + h, x:x + w])
        # on ne cherchera l'image qu'autour de l'endroit où elle apparaît (moins de fausses alertes)
        m = args.margin
        x0, y0 = max(0, x - m), max(0, y - m)
        region = [x0, y0, min(img.shape[1], x + w + m) - x0, min(img.shape[0], y + h + m) - y0]
        lst.append({"path": path, "region": region, "threshold": 0.8})
        save(cfg, args.config)
        print(f"Modèle enregistré : {path} (zone de recherche {region})")

    elif args.cmd == "point":
        # Donne les coordonnées des boutons (Jouer, Pause...) pour restart/pause/resume_sequence
        img = cv2.imread(args.image or "calib/screen.png")
        print("Cliquez sur un bouton : ses coordonnées s'affichent. ÉCHAP pour quitter.")

        def on_click(event, x, y, *_):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(f"  - {{click: [{x}, {y}]}}")
                cv2.circle(img, (x, y), 6, (0, 0, 255), 2)
                cv2.imshow("Points", img)

        cv2.imshow("Points", img)
        cv2.setMouseCallback("Points", on_click)
        while cv2.waitKey(50) != 27:
            pass
        cv2.destroyAllWindows()

    elif args.cmd == "digits":
        # Apprend la forme des chiffres du jeu : on montre chaque zone, vous tapez ce qui est affiché
        from subway_rl.glyphs import GROUPS, GlyphReader
        folder = cfg["game"].get("digits_folder", "templates/digits")
        readers = {grp: GlyphReader(folder, grp) for grp in set(GROUPS.values())}
        regions = {k: v for k, v in cfg["game"]["regions"].items() if k in GROUPS and v}
        print("Pour chaque zone, tapez EXACTEMENT ce qui est affiché (ex : 42 471, 74, x8,5),")
        print("Entrée vide = passer cette zone, q = terminer.\n")
        first = True
        while True:
            if args.image and first:
                img = cv2.imread(args.image)
            else:
                input("Lancez/continuez une partie, puis appuyez sur Entrée pour capturer (3 s)... ")
                time.sleep(3)
                img = grab(cfg)
            first = False
            for name, (x, y, w, h) in regions.items():
                roi = img[y:y + h, x:x + w]
                cv2.imshow("Que lisez-vous ?", cv2.resize(roi, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
                cv2.waitKey(300)
                txt = input(f"  {name} : ").strip()
                if txt.lower() == "q":
                    cv2.destroyAllWindows()
                    return
                if not txt:
                    continue
                r = readers[GROUPS[name]]
                if r.learn(roi, txt):
                    print(f"    appris ({r.mode})")
                else:
                    print("    impossible de découper les caractères : agrandissez un peu la zone "
                          f"(python calibrate.py region {name}) ou passez")
            cv2.destroyAllWindows()
            for grp, r in readers.items():
                miss = r.missing()
                names = " / ".join(k for k, v in GROUPS.items() if v == grp)
                print(f"  {names} : " + (f"il manque encore les chiffres {' '.join(miss)}" if miss
                                         else "tous les chiffres sont appris"))
            if all(not r.missing() for g_, r in readers.items() if g_ == "hud"):
                print("Score et pièces OK. Continuez si le multiplicateur manque des chiffres, sinon q.")

    elif args.cmd == "test":
        from subway_rl.hud import HudReader
        hud = HudReader(cfg["game"])
        print("Lecture du HUD en direct (Ctrl+C pour arrêter)")
        print("Les lectures ratées sont enregistrées dans calib/debug/ (envoyez-les si besoin).")
        os.makedirs("calib/debug", exist_ok=True)
        saved, i = {}, 0
        readers = {"score": hud.score, "coins": hud.coins, "multiplier": hud.mult}
        while True:
            img = grab(cfg)
            s = hud.read(img)
            i += 1
            extra = " ".join(f"{d.score(img):.2f}" for d in hud.death_detectors)
            extra += " | étoile: " + " ".join(f"{d.score(img):.2f}" for d in hud.star_detectors)
            pb = []
            for name, r in readers.items():
                if r is None or r.status in ("ok", "à confirmer"):
                    continue
                pb.append(f"{name} {r.status} (lu {r.last_raw})")
                if saved.get(name, 0) < 30:  # garde une image des lectures ratées
                    saved[name] = saved.get(name, 0) + 1
                    x, y, w, h = r.region
                    cv2.imwrite(f"calib/debug/{name}_{i:05d}_{r.status}_{r.last_raw}.png".replace(" ", ""),
                                img[y:y + h, x:x + w])
            print(f"score={s.score:>8} pièces={s.coins:>5} mult=x{s.multiplier:<5} étoiles={s.stars} "
                  f"mort={s.dead} choc={s.stumbled}  [similarité mort: {extra}]"
                  + (f"  !! {', '.join(pb)}" if pb else ""))
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
