# Subway Surfers City – Apprentissage par renforcement profond

Un agent qui apprend à jouer à **Subway Surfers City** (l'appli Google Play) **uniquement à partir
des pixels de l'écran**. On ne lui dit jamais ce qu'est un train, une barrière, une pièce ou une
étoile : il le découvre tout seul grâce aux récompenses.

Le jeu tourne sur votre **téléphone Android**, l'écran est recopié sur le **PC Windows** par
scrcpy, et c'est le PC qui regarde, décide et apprend :

```
 Téléphone (Subway Surfers City)
     │  image de l'écran ──────────►  PC : fenêtre scrcpy
     │                                   ▼
     │                              capture → réseau de neurones → décision
     │                              + apprentissage (PPO + curiosité)
     ◄── swipes tactiles ◄──── glissements de souris dans scrcpy
```

## Récompenses

| Événement                               | Récompense (par défaut) |
|-----------------------------------------|-------------------------|
| chaque point de score gagné             | `+0.01`                 |
| chaque pièce ramassée                   | `+1`                    |
| chaque étoile ramassée (+1 au multiplicateur de la partie) | `+5` |
| se prendre un mur / trébucher           | `-3`                    |
| mourir                                  | `-10` (fin de partie)   |

Tout se règle dans `config.yaml`, section `rewards`. Les étoiles sont comptées en lisant le
multiplicateur affiché (« x12 ») : chaque +1 = une étoile.

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
calibrate.py          configuration du vrai jeu (zones du HUD, images de référence, boutons, test)
lancer_scrcpy.bat     affiche le téléphone sur le PC
config.yaml           récompenses, hyperparamètres, réglages du jeu
subway_rl/
  rewards.py          calcul des récompenses à partir du HUD
  sim_env.py          mini-simulateur pour tester l'apprentissage sans le jeu
  screen_env.py       environnement du VRAI jeu (capture d'écran + commandes)
  hud.py              lecture du score/pièces/multiplicateur (OCR), détection mort/choc
  controls.py         capture d'écran, swipes souris (scrcpy/BlueStacks), clavier ou adb
  curiosity.py        curiosité RND
  callbacks.py        statistiques TensorBoard, pause du jeu pendant l'apprentissage
```

## Installation (PC Windows)

1. **Python 3.11 ou 3.12** (python.org, cocher « Add Python to PATH »), **Git** (git-scm.com).
2. **Tesseract OCR** (lecture des chiffres du score) : installeur « UB Mannheim »
   (github.com/UB-Mannheim/tesseract/wiki), puis ajouter `C:\Program Files\Tesseract-OCR` au PATH.
3. **scrcpy** pour Windows (github.com/Genymobile/scrcpy → Releases ; adb est inclus). Copiez
   `lancer_scrcpy.bat` dans son dossier.
4. Le projet :
   ```powershell
   git clone https://github.com/developpeur950-design/subway-surfer.git
   cd subway-surfer
   git checkout claude/youthful-albattani-m7ol59
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   Avec une carte NVIDIA, installez PyTorch avec CUDA (commande sur pytorch.org) : c'est
   beaucoup plus rapide.

## 1. Tester avec le simulateur (recommandé en premier)

```powershell
python train.py --env sim --timesteps 2000000 --n-envs 8
tensorboard --logdir runs        # courbes : partie/score, partie/pieces, partie/etoiles, ...
python play.py --env sim --model models/ppo_sim_final.zip --gif partie.gif
```

Le simulateur est une version simplifiée (vue de dessus, 3 voies) avec les mêmes règles de
récompense. Il vérifie en quelques dizaines de minutes que tout fonctionne sur votre PC.

## 2. Préparer le téléphone

1. Installez **Subway Surfers City** depuis Google Play. Utilisez de préférence un compte
   secondaire : le jeu est en ligne et l'automatisation peut être contraire à ses conditions
   d'utilisation.
2. Paramètres → À propos → appuyez 7 fois sur « Numéro de build », puis dans les Options pour
   les développeurs activez **Débogage USB**.
3. Branchez le téléphone en USB, acceptez l'autorisation, activez **Ne pas déranger**,
   baissez la luminosité et laissez-le en charge.
4. Lancez `lancer_scrcpy.bat` : l'écran du téléphone s'affiche en haut à gauche du PC.
   **Ne déplacez plus cette fenêtre.** (Sans câble : ajoutez `--tcpip`, mais l'agent réagira un
   peu plus lentement.)
5. Ouvrez Subway Surfers City et allez jusqu'à l'écran « toucher pour jouer ».

## 3. Calibrer (une seule fois)

Les commandes `screenshot` attendent 3 s. Dans les fenêtres de sélection : tracez un rectangle à
la souris puis ENTRÉE.

```powershell
# a) où est l'écran du téléphone dans la fenêtre scrcpy
python calibrate.py screenshot --full
python calibrate.py region capture --image calib/full.png

# b) pendant une partie : les chiffres du HUD (encadrez juste les chiffres)
python calibrate.py screenshot
python calibrate.py region score
python calibrate.py region coins
python calibrate.py region multiplier

# c) sur l'écran de fin de partie : un élément qui n'apparaît QUE là
python calibrate.py screenshot
python calibrate.py template death

# d) (recommandé) juste après un choc sur le flanc d'un train : le garde qui apparaît derrière
python calibrate.py screenshot
python calibrate.py template stumble

# e) positions des boutons (rejouer, pause, reprendre...) : cliquez dessus
python calibrate.py point --image calib/screen.png
```

Copiez les coordonnées affichées par `point` dans `config.yaml` (`restart_sequence`,
`pause_sequence`, `resume_sequence`). Prenez une capture de chaque écran concerné (écran de fin,
résultats, menu pause) pour relever tous les boutons. Un clic droit dans scrcpy = touche Retour.

Vérifiez ensuite :
```powershell
python calibrate.py test     # pendant une partie : score/pièces/multiplicateur corrects, mort=True à la fin
python calibrate.py keys     # le personnage doit aller à gauche, droite, sauter, rouler
```

Sans la capture `stumble`, la pénalité pour les chocs contre un mur ne s'applique jamais (la mort,
elle, est toujours détectée).

## 4. Entraîner

```powershell
python train.py --env screen --timesteps 1000000
```

- Le jeu est mis en pause (bouton pause) pendant chaque mise à jour du réseau.
- L'agent utilise la souris : n'y touchez pas. **Arrêt d'urgence** : souris dans un coin de
  l'écran, ou Ctrl+C (le modèle est sauvegardé).
- Sauvegardes automatiques dans `models/` ; pour reprendre :
  `python train.py --env screen --load models/ppo_screen_final.zip --timesteps 1000000`
- Suivi : `tensorboard --logdir runs` puis http://localhost:6006.
- Le regarder jouer : `python play.py --env screen --model models/ppo_screen_final.zip`

### Conseils

- Le jeu tourne en temps réel (~10 actions/s) : 1 million de pas ≈ 28 h de jeu. Laissez tourner
  la nuit, plusieurs nuits.
- S'il ne fait que ramasser des pièces et meurt vite : `death: -20`.
- S'il répète toujours la même action : augmentez `agent.ent_coef` (0.02) ou `curiosity.beta`.
- Une pub ou une récompense quotidienne bloque le jeu ? Ajoutez un `click` pour la fermer dans
  `restart_sequence`.
- `obs_crop_y` permet de retirer le HUD de l'image vue par l'agent.

### Autres façons de lancer le jeu

- **BlueStacks / LDPlayer** sur le PC (installe la vraie appli depuis Google Play) : même
  configuration, en encadrant la fenêtre de l'émulateur. Vous pouvez aussi associer les flèches du
  clavier à des swipes dans l'éditeur de contrôles de BlueStacks et mettre `controller: keyboard`.
- **Subway Surfers original** (navigateur) : `controller: keyboard` et des `key` dans les séquences.
