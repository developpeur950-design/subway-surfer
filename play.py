"""Faire jouer un agent entraîné (et éventuellement enregistrer une vidéo du simulateur).

    python play.py --env sim --model models/ppo_sim_final.zip --episodes 5 --gif partie.gif
    python play.py --env screen --model models/ppo_screen_final.zip
"""
from __future__ import annotations

import argparse

import numpy as np
from stable_baselines3 import PPO

from train import build_env, load_config, make_overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["sim", "screen"], default="sim")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--stochastic", action="store_true", help="échantillonner les actions au lieu de prendre la meilleure")
    ap.add_argument("--gif", help="(simulateur) enregistre la première partie dans ce fichier GIF")
    ap.add_argument("--overlay", action="store_true", help="fenêtre temps réel : cadres autour des éléments")
    ap.add_argument("--overlay-video", help="enregistre aussi cet affichage dans une vidéo .mp4")
    args = ap.parse_args()

    cfg = load_config(args.config)
    venv, _ = build_env(args.env, cfg, n_envs=1, seed=123)
    model = PPO.load(args.model, device="cpu")

    overlay = (make_overlay(cfg, args.env, args.overlay, args.overlay_video)
               if args.overlay or args.overlay_video else None)
    frames, done_eps, obs = [], 0, venv.reset()
    while done_eps < args.episodes:
        action, _ = model.predict(obs, deterministic=not args.stochastic)
        obs, _, dones, infos = venv.step(action)
        if overlay:
            overlay.update(venv.env_method("frame_bgr", indices=[0])[0], infos[0], model, obs, action[0])
        if args.gif and done_eps == 0:
            frames.append(venv.render())
        if dones[0]:
            info = infos[0]
            t = info.get("episode_totals", {})
            print(f"Partie {done_eps + 1}: score={info.get('score')} pièces={info.get('coins')} "
                  f"étoiles={info.get('stars')} récompense={sum(t.values()):.1f} "
                  f"(trébuchements={t.get('stumble', 0) / (cfg['rewards'].get('stumble') or 1):.0f})")
            done_eps += 1
    venv.close()
    if overlay:
        overlay.close()

    if args.gif and frames:
        import imageio.v2 as imageio
        frames = [np.kron(f, np.ones((4, 4, 1), np.uint8)) for f in frames if f is not None]
        imageio.mimsave(args.gif, frames, duration=1 / 15)
        print(f"GIF enregistré : {args.gif}")


if __name__ == "__main__":
    main()
