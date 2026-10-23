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
| chaque point de score gagné             | `+0.001`                |
| chaque pièce ramassée                   | `+1`                    |
| chaque étoile secrète ramassée          | `+5`                    |
| se prendre un mur / trébucher           | `-3`                    |
| mourir                                  | `-10` (fin de partie)   |

Tout se règle dans `config.yaml`, section `rewards`. Les étoiles sont détectées **visuellement**
grâce au bandeau « ÉTOILE SECRÈTE » qui s'affiche au milieu de l'écran quand on en ramasse une
(pas via le multiplicateur, qui peut changer pour d'autres raisons).

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
lancer_scrcpy.bat     affiche le téléphone sur le PC (double-clic)
scrcpy/               scrcpy v5.0.1 + adb pour Windows (licence Apache 2.0)
config.yaml           récompenses, hyperparamètres, réglages du jeu
subway_rl/
  rewards.py          calcul des récompenses à partir du HUD
  sim_env.py          mini-simulateur pour tester l'apprentissage sans le jeu
  screen_env.py       environnement du VRAI jeu (capture d'écran + commandes)
  hud.py              lecture du score/pièces/multiplicateur (OCR), détection mort/choc
  controls.py         capture d'écran, swipes souris (scrcpy/BlueStacks), clavier ou adb
  curiosity.py        curiosité RND
  callbacks.py        statistiques TensorBoard, pause du jeu pendant l'apprentissage
  overlay.py          cadres en temps réel : découverte des éléments, obstacles/bonus, attention
```

## Installation (PC Windows)

1. **Python 3.11 ou 3.12** (python.org, cocher « Add Python to PATH »), **Git** (git-scm.com).
2. **Tesseract OCR** (lecture des chiffres du score) : installeur « UB Mannheim »
   (github.com/UB-Mannheim/tesseract/wiki), dossier d'installation par défaut.
3. Le projet (**scrcpy et adb sont inclus** dans le dossier `scrcpy/`, rien à installer) :
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
3. Branchez le téléphone en USB. Dans la notification « Paramètres USB », choisissez
   **USB contrôlé par : Cet appareil** et **Transfert de fichiers / Android Auto**, puis acceptez
   l'autorisation, activez **Ne pas déranger**,
   baissez la luminosité et laissez-le en charge.
4. Double-cliquez sur `lancer_scrcpy.bat` (dans le dossier du projet) : l'écran du téléphone s'affiche en haut à gauche du PC.
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

# b2) apprendre la police des chiffres du jeu (indispensable pour bien lire le score) :
#     tapez ce que vous voyez dans chaque zone, sur plusieurs captures, jusqu'à ce que
#     tous les chiffres 0-9 soient appris
python calibrate.py digits

# c) sur l'écran de fin de partie : un élément qui n'apparaît QUE là
python calibrate.py screenshot
python calibrate.py template death

# d) (recommandé) juste après un choc sur le flanc d'un train : le garde (et son chien) qui
#    apparaît en bas à gauche de l'écran ; encadrez sa tête/casquette
python calibrate.py screenshot
python calibrate.py template stumble

# e) étoiles secrètes : enregistrez ~15 s de jeu en ramassant une étoile secrète, puis dans
#    calib/rec/ choisissez une image où le bandeau est affiché et encadrez UNIQUEMENT le texte
#    « ÉTOILE SECRÈTE » (pas le nom en dessous, qui change d'une étoile à l'autre)
python calibrate.py record --seconds 15
python calibrate.py template star --image calib/rec/0123.png
#    (refaites-le 2 ou 3 fois avec d'autres images : star_2.png, star_3.png...)

# f) positions des boutons (rejouer, pause, reprendre...) : cliquez dessus
python calibrate.py point --image calib/screen.png
```

Pour le multiplicateur, laissez de la marge autour de « x8,5 » : il devient plus large
(« x11,3 ») et change de couleur en cours de partie.

Copiez les coordonnées affichées par `point` dans `config.yaml` (`restart_sequence`,
`pause_sequence`, `resume_sequence`). Prenez une capture de chaque écran concerné (écran de fin,
résultats, menu pause) pour relever tous les boutons. Un clic droit dans scrcpy = touche Retour.

Vérifiez ensuite :
```powershell
python calibrate.py test     # pendant une partie : score/pièces corrects, étoiles +1 à chaque étoile, mort=True à la fin
python calibrate.py keys     # le personnage doit aller à gauche, droite, sauter, rouler
```

Pour l'étoile, n'encadrez que le texte « ÉTOILE SECRÈTE » : il est identique à chaque fois, alors
que le nom de l'étoile (« TOMBER AVEC PANACHE »…) et l'icône peuvent changer. Si `test` compte
des étoiles en trop, montez `threshold` (0.85–0.9) pour `star` dans `config.yaml` ; s'il n'en
compte pas, baissez-le (0.7) ou ajoutez une autre image.

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

### Voir ce que l'IA reconnaît (cadres en temps réel)

```powershell
python train.py --env screen --timesteps 1000000 --overlay
python play.py --env screen --model models/ppo_screen_final.zip --overlay
```

Une fenêtre s'ouvre à droite de scrcpy avec un cadre autour de chaque élément du jeu :

- **Personne n'a étiqueté les éléments** : l'IA repère ce qui se détache du décor et regroupe
  les éléments qui se ressemblent en types qu'elle invente (T0, T1, T2…).
- **Elle juge chaque type d'après ce qui se passe quand le personnage le touche** :
  - **OBSTACLE**, cadre rouge : le type est plus souvent suivi d'un choc ou d'une mort que la
    moyenne (trains, barrières…) ;
  - **BONUS**, cadre vert : le type est plus souvent suivi de pièces ou d'étoiles (pièces,
    bonus…) ;
  - **neutre** ou **?** : pas encore assez d'observations, ou aucun effet notable.
- **La carte de chaleur** (rouge/jaune) montre où le réseau qui joue regarde pour décider ;
  « att 60 % » indique l'attention qu'il porte à un élément.
- **Le panneau de droite** liste les types découverts avec leurs gains et dangers moyens.

Le **personnage** est suivi en temps réel (cadre blanc « PERSONNAGE ») : partout où il va,
y compris en l'air ou dans la bulle. Le suivi part de sa place habituelle en bas au centre ; pour
la régler précisément, pendant une partie :
```powershell
python calibrate.py screenshot
python calibrate.py region player      # encadrez le personnage
```

Ces étiquettes s'affinent au fil des parties. Les premières minutes, presque tout est « ? ».
Ajoutez `--overlay-video partie.mp4` pour enregistrer l'affichage. Réglages dans `config.yaml`,
section `overlay` : sensibilité, zone du personnage, seuils obstacle/bonus. Si le PC rame,
mettez `every_n_steps: 2`.

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
