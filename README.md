# Subway Surfers – Apprentissage par renforcement profond

Un agent qui apprend à jouer à Subway Surfers **uniquement à partir des pixels de l'écran**.
On ne lui dit jamais ce qu'est un train, une barrière, une pièce ou une étoile : il le
découvre tout seul grâce aux récompenses.

## Récompenses

| Événement                               | Récompense (par défaut) |
|-----------------------------------------|-------------------------|
| chaque point de score gagné             | `+0.01`                 |
| chaque pièce ramassée                   | `+1`                    |
| chaque étoile multiplicatrice ramassée  | `+5`                    |
| se prendre un mur / trébucher           | `-3`                    |
| mourir                                  | `-10` (fin de partie)   |

Tout se règle dans `config.yaml`, section `rewards`.

## Comment l'agent apprend et découvre seul

1. **Vision apprise (CNN)** : l'image du jeu (84×84 couleur, 4 images empilées pour percevoir le
   mouvement) passe dans un réseau convolutif. Les filtres ne sont pas programmés : ils apprennent
   d'eux-mêmes à détecter les éléments utiles (trains, barrières, pièces, étoiles...), parce que ça
   permet de gagner plus de récompense.
2. **PPO** (Proximal Policy Optimization, via Stable-Baselines3) : l'algorithme qui ajuste la
   politique (gauche / droite / saut / roulade / rien) pour maximiser la somme des récompenses.
3. **Curiosité (RND – Random Network Distillation)** : une récompense *intrinsèque* bonus quand
   l'agent voit quelque chose de nouveau. Il est ainsi poussé à explorer et à découvrir des
   situations qu'il ne connaît pas encore (power-ups, toits des trains, nouveaux décors…).
   Réglable avec `curiosity.beta` (0 = désactivée).

Le score, les pièces, le multiplicateur, la mort et les chocs sont lus à l'écran (OCR + images de
référence) **seulement pour calculer les récompenses** ; l'agent ne reçoit jamais ces informations
directement.

## Structure

```
train.py              entraînement (simulateur ou vrai jeu)
play.py               faire jouer un modèle entraîné
calibrate.py          configuration du vrai jeu (zones du HUD, images de référence, test des touches)
config.yaml           récompenses, hyperparamètres, réglages du jeu
subway_rl/
  rewards.py          calcul des récompenses à partir du HUD
  sim_env.py          mini-simulateur pour tester l'apprentissage sans le jeu
  screen_env.py       environnement du VRAI jeu (capture d'écran + commandes)
  hud.py              lecture du score/pièces/multiplicateur (OCR), détection mort/choc
  controls.py         capture d'écran, clavier (navigateur/PC) ou adb (Android/BlueStacks)
  curiosity.py        curiosité RND
  callbacks.py        statistiques TensorBoard, pause du jeu pendant l'apprentissage
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Pour le vrai jeu il faut aussi :
- **Tesseract OCR** (lecture des chiffres) : Windows → installeur « UB Mannheim », Linux →
  `sudo apt install tesseract-ocr`, macOS → `brew install tesseract`.
- **adb** (seulement si vous jouez sur Android/BlueStacks).

Une carte graphique NVIDIA accélère beaucoup l'apprentissage, mais ce n'est pas obligatoire.

## 1. Tester avec le simulateur (recommandé en premier)

```bash
python train.py --env sim --timesteps 2000000 --n-envs 8
tensorboard --logdir runs        # courbes : partie/score, partie/coins, partie/stars, ...
python play.py --env sim --model models/ppo_sim_final.zip --gif partie.gif
```

Le simulateur est une version simplifiée (vue de dessus, 3 voies) avec les mêmes règles de
récompense. Il permet de vérifier en quelques dizaines de minutes que tout fonctionne.

## 2. Brancher le vrai jeu

1. Ouvrez Subway Surfers (version navigateur, PC ou émulateur Android) dans une fenêtre fixe.
2. Trouvez la zone du jeu :
   ```bash
   python calibrate.py screenshot --full
   python calibrate.py region capture --image calib/full.png
   ```
3. Lancez une partie, faites une capture, puis sélectionnez les zones du HUD :
   ```bash
   python calibrate.py screenshot
   python calibrate.py region score
   python calibrate.py region coins
   python calibrate.py region multiplier
   ```
4. Mourez volontairement, capturez l'écran de mort et découpez-y un élément caractéristique
   (ex. le bouton « Rejouer ») :
   ```bash
   python calibrate.py screenshot
   python calibrate.py template death
   ```
   Faites de même pour un choc contre un mur (`template stumble`, par exemple l'inspecteur qui
   apparaît derrière le personnage) et éventuellement l'icône du bonus multiplicateur
   (`template star`).
5. Vérifiez la lecture et les commandes :
   ```bash
   python calibrate.py test     # le score/les pièces doivent s'afficher correctement
   python calibrate.py keys
   ```
6. Adaptez dans `config.yaml` les séquences `restart_sequence`, `pause_sequence`,
   `resume_sequence` à votre version du jeu (touches ou clics pour relancer une partie).
7. Entraînez :
   ```bash
   python train.py --env screen --timesteps 1000000
   # partir du modèle du simulateur n'est pas utile (images trop différentes) : partez de zéro
   ```
   Le jeu est mis en pause automatiquement pendant chaque mise à jour du réseau.
   Arrêt d'urgence (clavier) : déplacez la souris dans un coin de l'écran.

### Conseils

- Le vrai jeu tourne en temps réel (~10 actions/s) : comptez plusieurs heures/jours pour de bons
  résultats. Laissez tourner et reprenez avec `--load models/ppo_screen_XXXX_steps.zip`.
- Si l'agent ne fait que ramasser des pièces et meurt vite, augmentez la pénalité `death`.
- Si l'agent n'explore pas assez, augmentez `curiosity.beta` ou `agent.ent_coef`.
- `obs_crop_y` permet de retirer le HUD de l'image vue par l'agent.
- Respectez les conditions d'utilisation du jeu : n'utilisez l'agent que hors ligne / en local,
  pas pour des classements en ligne.
