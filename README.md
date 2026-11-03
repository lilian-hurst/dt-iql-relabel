# Decision Transformer + relabellisation des RTG par IQL

Projet final *AI for Robotics — Track 1* : reproduction de Decision Transformer (Chen et al., 2021),
puis amélioration de sa capacité de *stitching* en relabellisant les return-to-go avec un critique
IQL (Kostrikov et al., 2022), en comparaison avec QDT (Yamagata et al., 2023).
Voir `final_project_proposal.pdf`.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                 # ou : pip install -r requirements.txt
pip install -e ".[kitchen]"             # optionnel : Franka Kitchen (Minari + Gymnasium-Robotics)
python scripts/check_env.py             # vérifie dépendances, env MuJoCo, dataset D4RL
pytest -q                               # tests hors-ligne du pipeline de données
```

## Données

Ce sont les vrais datasets D4RL, ceux du papier DT :

| Tâches | Source | Évaluation |
|---|---|---|
| halfcheetah / hopper / walker2d × medium, medium-replay, medium-expert | HDF5 officiels D4RL v2 : serveur du RAIL (Berkeley), sinon copie identique sur Hugging Face ([`imone/D4RL`](https://huggingface.co/datasets/imone/D4RL)) | Gymnasium `*-v4`, MuJoCo 3 |
| kitchen-complete / partial / mixed | Minari [`D4RL/kitchen/*-v2`](https://minari.farama.org/datasets/D4RL/kitchen/) (Hugging Face `farama-minari`) | `FrankaKitchen-v1`, épisodes de 450 pas |

```bash
python scripts/download_datasets.py                     # 9 datasets Gym-MuJoCo, avec résumé de vérification
python scripts/download_datasets.py --suite kitchen     # Franka Kitchen (pip install -e ".[kitchen]")
python scripts/check_env.py --dataset hopper-medium-replay-v2   # test de rejeu de la dynamique
```

Les sources sont essayées dans l'ordre de `download_sources` (`config/data.yaml`). Chaque fichier est
vérifié (lecture HDF5, clés, longueurs cohérentes) avant d'être conservé dans `datasets/`
(ou `$D4RL_DATASET_DIR`). Si les deux sources sont indisponibles, déposez les fichiers à la main dans ce
dossier ou passez `--file`.

Le paquet `d4rl` (mujoco-py) n'est pas utilisé : il est difficile à installer aujourd'hui.
`check_env.py` rejoue des transitions du dataset dans l'environnement (via qpos/qvel) pour vérifier que
la dynamique de MuJoCo 3 correspond bien aux données.

Le miroir [`edbeeching/decision_transformer_gym_replay`](https://huggingface.co/datasets/edbeeching/decision_transformer_gym_replay)
(tutoriel DT de Hugging Face) n'est pas utilisé : il n'a ni medium-expert, ni timeouts, ni qpos/qvel.
Les datasets Minari PointMaze ne remplacent pas Maze2D : leur but change à chaque épisode, ce qui
demanderait une politique conditionnée par le but.

Le score reporté est le score normalisé D4RL `100·(R − R_random)/(R_expert − R_random)` (`src/evaluation/normalization.py`).

## Entraîner sur un Mac Apple Silicon (M1 à M4)

```bash
# Python 3.11 ou 3.12 natif arm64 (uv, Homebrew ou Miniforge), puis :
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,kitchen]"                       # PyTorch macOS avec GPU Metal (MPS)
python scripts/download_datasets.py --suite mujoco kitchen
pytest -q && python scripts/check_env.py
python scripts/benchmark_device.py --write            # cpu ou mps, par algorithme, mesuré sur la machine
python experiments/run_experiments.py experiments/configs/smoke.yaml          # chaîne complète, minutes
python experiments/run_experiments.py experiments/configs/laptop.yaml --dry-run --jobs 4   # durée estimée
caffeinate -i python experiments/run_experiments.py experiments/configs/laptop.yaml --jobs 4
caffeinate -i python experiments/run_ablations.py --config experiments/configs/ablations_laptop.yaml --jobs 4
```

- **Device :** `device: auto` choisit MPS sur Mac. Mais pour de si petits réseaux, le CPU peut être plus
  rapide (surtout pour IQL). `benchmark_device.py --write` mesure les deux et écrit le meilleur choix par
  algorithme, avec les débits, dans `config/experiment.local.yaml` (non versionné). On peut forcer le
  device avec `--device cpu|mps` ou `DTIQL_DEVICE`.
- **Parallélisme :** avec `--jobs N`, les cœurs CPU sont répartis automatiquement entre les jobs
  (`DTIQL_THREADS`). Les opérations non disponibles sur MPS retombent sur le CPU
  (`PYTORCH_ENABLE_MPS_FALLBACK=1`, fixé par le paquet).
- **Budget :** `laptop.yaml` reproduit DT sur les 9 datasets (3 seeds, 100k pas comme le papier) et
  limite les méthodes avec critique aux 3 datasets medium-replay, avec des critiques à 300k pas et IQL à
  500k. Cela fait 108 jobs. `ablations_laptop.yaml` réutilise ses runs : seuls 108 jobs de plus.
  `--dry-run` affiche la durée estimée à partir des débits mesurés.
- **Pratique :** branchez le Mac sur secteur ; `caffeinate -i` empêche la mise en veille. Un run
  interrompu repart de zéro, mais l'expérience reprend après les runs déjà terminés.

## Structure

```
config/           data, dt, iql, cql, relabel, methods, experiment (.yaml)
src/utils/        config (YAML + surcharges `clé=valeur`), graines/device, logger
src/data/         registre des tâches, chargement D4RL/Minari, fenêtres de K pas, règles de relabellisation
src/envs/         environnements d'évaluation (+ correctif Kitchen)
src/models/       DecisionTransformer, IQL, CQL
src/training/     train_dt, train_offline (IQL/CQL), relabel_returns
src/evaluation/   score normalisé, rollouts DT et politiques, ré-évaluation de checkpoints
experiments/      lanceur, expériences, ablations, agrégation, figures ; configs/ : main, stitching,
                  kitchen, smoke, ablations, ablations_smoke
scripts/          check_env.py
tests/            tests unitaires (ceux qui demandent PyTorch sont ignorés s'il est absent)
```

## Phase 1 — Decision Transformer

```bash
python -m src.training.train_dt --dataset hopper-medium-replay-v2 --seed 0
python -m src.training.train_dt --dataset hopper-medium-v2 --set model.context_len=5   # surcharge
python -m src.evaluation.eval_dt runs/dt/hopper-medium-v2/seed0/model.pt --targets 3600 1800
```

- Modèle (`src/models/decision_transformer.py`) : tokens (R̂, s, a), embedding de pas de temps, GPT causal pré-LN,
  action prédite depuis le token d'état ; hyperparamètres Gym du papier (K=20, d=128, 3 couches, 1 tête).
- Données (`src/data/sequence_dataset.py`) : fenêtres de K pas tirées comme dans le code officiel,
  padding à gauche + masque, états normalisés, RTG / 1000.
- Entraînement (`src/training/train_dt.py`) : MSE sur les actions, AdamW 1e-4, warmup 10k, clip 0.25, 100k pas.
- Évaluation (`src/evaluation/rollout_dt.py`) : N épisodes en parallèle, R̂ décrémenté de la récompense,
  score normalisé D4RL ; évaluation finale sur 100 épisodes par return cible, la meilleure cible est reportée.

## Phase 2 — IQL et relabellisation des RTG

```bash
# baseline IQL (τ=0.7, β=3, 1M pas, politique AWR)
python -m src.training.train_iql --dataset hopper-medium-replay-v2 --seed 0
# critique de relabellisation (γ=0.995, t/H en entrée) + relabellisation, en une commande
python -m src.training.relabel_returns --dataset hopper-medium-replay-v2 --seed 0 --train-critic
python -m src.training.relabel_returns --dataset hopper-medium-replay-v2 --set rule=qdt alpha=0.5
# IQL-DT : DT entraîné sur les RTG relabellisés
python -m src.training.train_dt --dataset hopper-medium-replay-v2 --method iql_dt \
    --rtg-file runs/relabel/iql_max_q_a1/hopper-medium-replay-v2/seed0/rtg.npz
```

Règles (`src/data/relabel.py`, détails dans `config/relabel.yaml`) :

- `max_q` : R̂_t = R_t + α·max(0, Q(s_t,a_t) − G^γ_t). Avec γ=1 et α=1, on obtient **max(R_t, Q(s_t,a_t))**.
- `qdt` : récursion arrière R̂_t = r_t + R̂_{t+1} + α·max(0, V(s_{t+1}) − Ĝ_{t+1}), soit la règle QDT avec γ=1.

Les RTG du DT ne sont pas actualisés, alors que Q l'est. Le gain est donc mesuré face au return
Monte-Carlo actualisé avec le même γ que le critique. Ce critique est entraîné avec γ proche de 1 et t/H en entrée
(valeur à horizon fini), et ses valeurs sont remises à l'échelle des récompenses brutes.
`stats.json` indique la part des pas relabellisés, le gain moyen et le biais du critique face à Monte-Carlo.

## Expériences et comparaison

```bash
python experiments/run_experiments.py experiments/configs/smoke.yaml               # chaîne complète, ~minutes
python experiments/run_experiments.py experiments/configs/main.yaml --dry-run      # 450 jobs planifiés
python experiments/run_experiments.py experiments/configs/main.yaml --jobs 4 --gpus 0 1
python experiments/run_experiments.py experiments/configs/stitching.yaml --jobs 4  # meilleure moitié retirée
python experiments/run_experiments.py experiments/configs/kitchen.yaml --jobs 4    # Franka Kitchen
python experiments/aggregate.py runs/main                                          # tableau seul
```

Méthodes (`config/methods.yaml`) :

| Méthode | Critique | Règle de relabellisation |
|---|---|---|
| `dt` | – | – (RTG Monte-Carlo) |
| `iql`, `cql` | – | – (baselines offline RL) |
| `qdt` | CQL | récursive (QDT) |
| `iql_dt` (proposée) | IQL | max(R_t, Q) |
| `iql_qdt` (contrôle) | IQL | récursive (QDT) |

Comparer `iql_qdt` à `qdt` isole l'effet du critique, et `iql_dt` à `iql_qdt` celui de la règle.
Le lanceur entraîne les critiques, relabellise, puis entraîne les DT. Il dédoublonne les critiques partagés,
saute les jobs déjà terminés (reprise après interruption) et écrit les logs dans `runs/<nom>/logs/`.
Les tableaux (moyenne ± écart-type sur les seeds, avec les scores publiés à côté) vont dans
`results/<nom>/table.{md,tex}`.

**Franka Kitchen** : données Minari `D4RL/kitchen/*-v2` (téléchargées via Hugging Face) et environnement
`FrankaKitchen-v1`. `src/envs/compat.py` corrige une incompatibilité entre gymnasium-robotics ≤ 1.4.2 et
MuJoCo ≥ 3.3 qui fait planter `reset()`.

## Phase 3 — Ablations et figures

```bash
python experiments/run_ablations.py --config experiments/configs/ablations_smoke.yaml  # chaîne complète, ~minutes
python experiments/run_ablations.py --dry-run                                          # plan et coût
python experiments/run_ablations.py --only context_len relabel_strength --jobs 4 --gpus 0 1
python experiments/plot_results.py experiment runs/main     # figures d'une expérience
python experiments/plot_results.py ablations                # tableaux + figures des ablations lancées
```

| Ablation | Paramètre | Valeurs | Méthodes (références en pointillés) |
|---|---|---|---|
| `context_len` | longueur de contexte K | 1, 5, 20 | DT, IQL-DT (IQL) |
| `relabel_strength` | α | 0, 0.25, 0.5, 1 | IQL-DT, QDT (DT, IQL) |
| `q_threshold` | gain minimal (fraction de \|G\|) | 0, 0.05, 0.1, 0.25 | IQL-DT (DT) |
| `critic_discount` | γ du critique | 0.99, 0.995, 0.999, 1 (règle exacte max(R_t, Q)) | IQL-DT (DT) |
| `dataset_quality` | quantile de return conservé | 0.25, 0.5, 0.75, 1 | DT, IQL, QDT, IQL-DT |

α = 0 laisse les RTG intacts : IQL-DT doit alors retrouver le score de DT, ce qui sert de contrôle de
cohérence. Les runs à la configuration par défaut (K=20, α=1, seuil 0, γ=0.995, quantile 1) sont communs
à toutes les ablations. Les critiques et relabellisations sont réutilisés quand le paramètre ne les
change pas : l'ensemble représente 567 jobs au lieu de 873 sans partage (3 datasets × 3 seeds, voir `--dry-run`).

Figures (PDF à texte éditable + PNG 300 dpi, dans `figures/`) :

- `comparison` : score par dataset et moyenne (point ± écart-type sur les seeds), score DT publié en repère.
- `learning_curves` : moyenne et bande ± écart-type pendant l'entraînement ; baselines en tirets.
- `relabel/<règle>__<dataset>` : calibration du critique (Q contre return Monte-Carlo), distribution des gains,
  exemple de trajectoire, RTG initiaux avant et après.
- `ablations/<ablation>` : score en fonction du paramètre, un panneau par dataset plus la moyenne. Pour
  `dataset_quality`, la courbe pointillée donne le score de la meilleure trajectoire conservée : la dépasser,
  c'est assembler des morceaux de trajectoires (stitching).

L'écart-type est calculé sur les seeds (ddof=0). La « Moyenne » est d'abord moyennée sur les datasets
pour chaque seed, puis résumée par moyenne ± écart-type. Chaque méthode garde la même couleur et la même
forme de marqueur dans toutes les figures (palette vérifiée pour le daltonisme), et chaque figure a son
tableau équivalent (`results/…/table.{md,tex}`).

## Configuration

Chaque script charge un YAML de `config/` puis accepte des surcharges :
`--set model.context_len=5 train.lr=3e-4`. Les sections `per_domain` (ex. IQL sur Kitchen)
sont appliquées automatiquement selon le dataset. Toute clé inconnue lève une erreur, ce qui évite
qu'une faute de frappe passe inaperçue dans une ablation.
