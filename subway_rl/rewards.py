"""Calcul des récompenses à partir de l'état du HUD (score, pièces, multiplicateur...).

Les deux environnements (simulateur et vrai jeu) produisent un ``HudState`` à
chaque pas ; ``RewardTracker`` transforme la différence entre deux états
successifs en récompense :

    + per_point            pour chaque point de score gagné
    + per_coin             pour chaque pièce ramassée
    + per_multiplier_star  pour chaque étoile multiplicatrice ramassée
    + death                (négatif) quand le personnage meurt
    + stumble              (négatif) quand il se prend un mur / trébuche
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields


@dataclass
class HudState:
    score: int = 0
    coins: int = 0
    multiplier: float = 1.0  # ex : 8.5 pour « x8,5 »
    stars: int = 0          # nombre total d'étoiles (secrètes) ramassées pendant la partie
    dead: bool = False
    stumbled: bool = False  # vrai uniquement au pas où le choc contre un mur se produit


@dataclass
class RewardConfig:
    per_point: float = 0.01
    per_coin: float = 1.0
    per_multiplier_star: float = 5.0
    death: float = -10.0
    stumble: float = -3.0
    alive_bonus: float = 0.0
    # Le score est multiplié par le multiplicateur du jeu : on peut le diviser
    # pour que l'agent ne soit pas récompensé "deux fois" pour une étoile.
    normalize_score_by_multiplier: bool = False
    # Garde-fou contre les erreurs d'OCR : un saut de score plus grand est ignoré.
    max_points_per_step: int = 50000

    @classmethod
    def from_dict(cls, d: dict | None) -> "RewardConfig":
        d = d or {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class RewardTracker:
    cfg: RewardConfig = field(default_factory=RewardConfig)
    prev: HudState = field(default_factory=HudState)
    totals: dict = field(default_factory=dict)

    def reset(self, state: HudState) -> None:
        self.prev = state
        self.totals = {"points": 0.0, "coins": 0.0, "stars": 0.0,
                       "death": 0.0, "stumble": 0.0, "alive": 0.0}

    def step(self, state: HudState) -> tuple[float, dict]:
        cfg, prev = self.cfg, self.prev

        d_score = state.score - prev.score
        if d_score < 0 or d_score > cfg.max_points_per_step:
            d_score = 0
        if cfg.normalize_score_by_multiplier and state.multiplier > 0:
            d_score = d_score / state.multiplier

        parts = {
            "points": cfg.per_point * d_score,
            "coins": cfg.per_coin * max(0, state.coins - prev.coins),
            "stars": cfg.per_multiplier_star * max(0, state.stars - prev.stars),
            "death": cfg.death if state.dead else 0.0,
            "stumble": cfg.stumble if state.stumbled else 0.0,
            "alive": 0.0 if state.dead else cfg.alive_bonus,
        }
        for k, v in parts.items():
            self.totals[k] += v
        self.prev = state
        return float(sum(parts.values())), parts
