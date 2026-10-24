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
| chaque seconde qui passe                | `-0.1`                  |

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
calibrate.py          configuration du vrai jeu (zones du HUD, police des chiffres, images de référence, boutons, personnage, test)
train_player.py       entraîne le détecteur du personnage
train_classifier.py   entraîne un détecteur oui/non (le garde, pour les chocs)
pretrained/           détecteurs (personnage, garde) et police des chiffres appris sur de vraies parties
lancer_scrcpy.bat     affiche le téléphone sur le PC (double-clic) ; lancer_scrcpy_wifi.bat : sans câble
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

### Sans câble (Wi-Fi)

Le PC et le téléphone doivent être sur le même réseau Wi-Fi. L'image arrive un peu plus tard
qu'en USB (l'agent réagit un peu moins vite) : préférez une box proche, en 5 GHz.

- **Méthode simple** : branchez le câble une fois, double-cliquez sur `lancer_scrcpy_wifi.bat`,
  puis débranchez quand la fenêtre du téléphone apparaît.
- **Sans jamais brancher** (Android 11 ou plus) : Options pour les développeurs → **Débogage
  sans fil** → activer → « Associer l'appareil avec un code d'association ». Sur le PC :
  ```powershell
  .\scrcpy\adb.exe pair 192.168.1.42:37123      # adresse et port affichés dans la fenêtre d'association, puis le code
  ```
  Puis écrivez l'adresse affichée sous « Adresse IP et port » (sur l'écran Débogage sans fil,
  ce n'est pas le même port) dans le fichier `calib\ip_telephone.txt`, par ex. `192.168.1.42:41567`,
  et lancez `lancer_scrcpy_wifi.bat`. Ce port change à chaque redémarrage du débogage sans fil.

Laissez le téléphone **sur son chargeur secteur** : l'entraînement dure des heures.

## 3. Calibrer (une seule fois)

Vos réglages sont enregistrés dans **`config.local.yaml`** : les mises à jour du projet
(`git pull`) ne le touchent jamais. Les commandes `screenshot` et `record` attendent 3 s.
Dans les fenêtres de sélection : tracez un rectangle à la souris puis ENTRÉE.

```powershell
# a) où est l'écran du téléphone dans la fenêtre scrcpy
python calibrate.py screenshot --full
python calibrate.py region capture --image calib/full.png

# b) pendant une partie : les zones du HUD (encadrez les chiffres avec un peu de marge)
python calibrate.py screenshot
python calibrate.py region score
python calibrate.py region coins          # le nombre seul, SANS l'icône de la pièce
python calibrate.py region multiplier     # « x8,5 » avec de la marge (il grossit et change de couleur)
python calibrate.py digits                # apprend la police des chiffres (plusieurs captures)

# c) « en jeu » : le bouton pause jaune en haut à gauche (sur la même capture de partie)
python calibrate.py template ingame
```

**Relance automatique des parties.** Le programme reconnaît les boutons des écrans qui suivent
une mort et clique dessus jusqu'à revenir en jeu. Capturez chaque écran (avec `record`, puis
choisissez l'image dans `calib/rec/`) et encadrez le bouton :

```powershell
python calibrate.py record --seconds 40        # mourez, puis laissez défiler jusqu'au menu
python calibrate.py template terminer --image calib/rec/XXXX.png   # « APPUIE POUR TERMINER LA COURSE »
python calibrate.py template continuer --image calib/rec/XXXX.png  # bouton orange CONTINUER (scores)
python calibrate.py template course --image calib/rec/XXXX.png     # bouton COURSE du menu
```

⚠️ Sur l'écran « Continuer la course ? », n'encadrez **jamais** le bouton qui dépense des clés :
encadrez le texte « APPUIE POUR TERMINER LA COURSE ». Si une fenêtre surprise bloque (pub,
récompense), capturez sa croix : `python calibrate.py template fermer --image ...`.
Pour la pause pendant l'apprentissage, capturez aussi le bouton pour reprendre :
`python calibrate.py template reprendre --image ...` (le bouton pause = image `ingame`).

**Étoiles et chocs** (sans eux, ces récompenses ne sont jamais données) :

```powershell
python calibrate.py record --seconds 20        # ramassez une étoile secrète
python calibrate.py template star --image calib/rec/XXXX.png       # le texte « ÉTOILE SECRÈTE » seul
```

**Chocs.** Un choc est compté quand les petites étoiles jaunes apparaissent autour de la tête
du personnage (il est étourdi ~3 s après chaque choc). Elles sont reconnues par un petit réseau
déjà entraîné sur trois parties (`pretrained/etourdi_classifier.pt`), qui regarde la zone de la
tête repérée par le détecteur du personnage. Si des chocs sont ratés ou inventés sur VOS parties :
```powershell
python calibrate.py record --seconds 90           # cognez-vous plusieurs fois
python calibrate.py label etourdi --every 3      # o = étoiles autour de la tête, n = non
python train_classifier.py etourdi               # crée models/etourdi_classifier.pt
```

Vérifiez en jouant :
```powershell
python calibrate.py test     # score, pièces, étoiles, choc (étourdi: 0.xx), en_jeu=True pendant la partie, mort=True à la fin
python calibrate.py keys     # le personnage doit aller à gauche, droite, sauter, rouler
```

Si vous aviez calibré avec une ancienne version (réglages dans `config.yaml`), récupérez-les :
```powershell
copy config.yaml config.ancien.yaml
git checkout -- config.yaml
git pull origin claude/youthful-albattani-m7ol59
python migrer_config.py config.ancien.yaml
```

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

Le **personnage** est retrouvé à chaque image (cadre blanc « PERSONNAGE ») par un petit
détecteur. Celui fourni dans `pretrained/` a été entraîné sur une vraie partie avec la tenue
actuelle du personnage. Si vous changez de personnage ou de tenue, ou s'il le rate souvent,
entraînez le vôtre (une trentaine d'images à encadrer, puis quelques minutes) :
```powershell
python calibrate.py record --seconds 60      # jouez : sauts, hoverboard, bonus, toits des trains...
python calibrate.py player                   # encadrez le personnage sur les images proposées
python train_player.py                       # crée models/player_detector.pt
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
