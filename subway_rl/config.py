"""Chargement de la configuration.

- ``config.yaml``        : réglages par défaut, mis à jour avec le projet (git pull).
- ``config.local.yaml``  : VOS réglages (calibration : zones, images de référence, boutons...).
  Il n'est jamais modifié par une mise à jour ; ses valeurs remplacent celles de config.yaml.
"""
from __future__ import annotations

import copy
import os

import yaml


def local_path(path: str = "config.yaml") -> str:
    root, ext = os.path.splitext(path)
    return f"{root}.local{ext}"


def _read(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str = "config.yaml") -> dict:
    return deep_merge(_read(path), _read(local_path(path)))


def load_local(path: str = "config.yaml") -> dict:
    return _read(local_path(path))


def save_local(local: dict, path: str = "config.yaml") -> str:
    p = local_path(path)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(local, f, allow_unicode=True, sort_keys=False)
    return p
