"""Entraînement de l'agent Subway Surfers (PPO + réseau convolutif + curiosité RND).

Exemples :
    python train.py --env sim --timesteps 2000000          # simulateur, pour tester
    python train.py --env screen --timesteps 1000000        # vrai jeu (après calibrate.py)
    python train.py --env screen --load models/ppo_screen_final.zip   # reprendre
"""
from __future__ import annotations

import argparse
import os

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

from subway_rl.callbacks import LiveOverlayCallback, PauseGameDuringUpdate, RewardBreakdownCallback
from subway_rl.curiosity import VecRND


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_env(kind: str, cfg: dict, n_envs: int, seed: int = 0):
    if kind == "sim":
        from subway_rl.sim_env import SubwaySimEnv
        venv = make_vec_env(SubwaySimEnv, n_envs=n_envs, seed=seed,
                            env_kwargs={"reward_config": cfg.get("rewards")},
                            vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv)
    else:
        from subway_rl.screen_env import SubwayScreenEnv
        # un seul jeu à l'écran -> un seul environnement
        venv = make_vec_env(SubwayScreenEnv, n_envs=1, seed=seed, env_kwargs={"cfg": cfg})
    channels = venv.observation_space.shape[-1]
    venv = VecFrameStack(venv, n_stack=cfg["agent"].get("frame_stack", 4))
    return venv, channels


def wrap_curiosity(venv, channels: int, cfg: dict):
    c = cfg.get("curiosity", {})
    if not c.get("enabled", True):
        return venv
    return VecRND(venv, frame_channels=channels, beta=c.get("beta", 0.5),
                  lr=c.get("lr", 1e-4), update_every=c.get("update_every", 256))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["sim", "screen"], default="sim")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--n-envs", type=int, default=8, help="(simulateur uniquement)")
    ap.add_argument("--load", help="modèle .zip à reprendre")
    ap.add_argument("--no-curiosity", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--overlay", action="store_true", help="fenêtre temps réel : cadres autour des éléments")
    ap.add_argument("--overlay-video", help="enregistre aussi cet affichage dans une vidéo .mp4")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.no_curiosity:
        cfg.setdefault("curiosity", {})["enabled"] = False
    a = cfg["agent"]

    venv, channels = build_env(args.env, cfg, args.n_envs if args.env == "sim" else 1)
    venv = wrap_curiosity(venv, channels, cfg)

    os.makedirs("models", exist_ok=True)
    tb = "runs" if _has_tensorboard() else None
    if args.load:
        model = PPO.load(args.load, env=venv, device=args.device, tensorboard_log=tb)
    else:
        model = PPO(
            "CnnPolicy", venv, device=args.device, tensorboard_log=tb, verbose=1,
            learning_rate=a.get("learning_rate", 2.5e-4),
            n_steps=a.get("n_steps", 256) if args.env == "sim" else a.get("n_steps_screen", 512),
            batch_size=a.get("batch_size", 256), n_epochs=a.get("n_epochs", 4),
            gamma=a.get("gamma", 0.99), gae_lambda=a.get("gae_lambda", 0.95),
            clip_range=a.get("clip_range", 0.1), ent_coef=a.get("ent_coef", 0.01),
            vf_coef=0.5, max_grad_norm=0.5,
        )

    callbacks = [RewardBreakdownCallback(),
                 CheckpointCallback(save_freq=max(1, a.get("save_every", 50_000) // venv.num_envs),
                                    save_path="models", name_prefix=f"ppo_{args.env}")]
    if args.env == "screen":
        game = venv.unwrapped.envs[0].unwrapped if hasattr(venv.unwrapped, "envs") else None
        if game is not None and cfg["game"].get("pause_sequence"):
            callbacks.append(PauseGameDuringUpdate(game))

    if args.overlay or args.overlay_video:
        callbacks.append(LiveOverlayCallback(make_overlay(cfg, args.env, args.overlay, args.overlay_video)))

    try:
        model.learn(total_timesteps=args.timesteps, callback=CallbackList(callbacks),
                    tb_log_name=f"ppo_{args.env}", reset_num_timesteps=not args.load)
    finally:
        path = f"models/ppo_{args.env}_final"
        model.save(path)
        print(f"Modèle sauvegardé : {path}.zip")
        venv.close()


def make_overlay(cfg: dict, env_kind: str, show: bool = True, video: str | None = None):
    from subway_rl.overlay import LiveOverlay
    ocfg = dict(cfg.get("overlay", {}))
    g = cfg.get("game", {})
    pos, crop = None, (0.0, 1.0)
    if env_kind == "screen":
        c = g["capture"]
        pos = (c["left"] + c["width"] + 10, max(0, c["top"] - 30))  # à droite de scrcpy
        crop = tuple(g.get("obs_crop_y", [0.0, 1.0]))
    else:
        ocfg = {**ocfg, "display_height": 420,
                "discovery": {**ocfg.get("discovery", {}), "ignore_top": 0.0, "roi": None,
                              "player_zone": [0.0, 0.85, 1.0, 1.0]}}
    return LiveOverlay(ocfg, window_pos=pos, video_path=video, show=show, crop_y=crop)


def _has_tensorboard() -> bool:
    try:
        import tensorboard  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
