"""Registre des tâches : nom D4RL -> source des données, environnement d'évaluation, scores de référence.

Les scores de référence sont ceux de d4rl/infos.py (REF_MIN_SCORE / REF_MAX_SCORE),
utilisés pour le score normalisé  100 * (R - R_random) / (R_expert - R_random).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Sources des fichiers HDF5 D4RL v2, essayées dans l'ordre (voir config/data.yaml : download_sources).
#   official : serveur du RAIL (Berkeley), source d'origine utilisée par le papier DT ;
#   hf_mirror : copie des mêmes fichiers HDF5, mêmes noms, sur Hugging Face (dataset imone/D4RL).
D4RL_HDF5_SOURCES = {
    "official": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/{filename}",
    "hf_mirror": "https://huggingface.co/datasets/imone/D4RL/resolve/main/{filename}",
}

MUJOCO_ENVS = {
    # domaine : (id Gymnasium, score random, score expert)
    "halfcheetah": ("HalfCheetah-v4", -280.178953, 12135.0),
    "hopper": ("Hopper-v4", -20.272305, 3234.3),
    "walker2d": ("Walker2d-v4", 1.629008, 4592.3),
}
MUJOCO_QUALITIES = ("random", "medium", "medium-replay", "medium-expert", "expert")

KITCHEN_TASKS = {
    # qualité : tâches à accomplir (D4RL kitchen-*-v0 / Minari D4RL/kitchen/*-v2)
    "complete": ["microwave", "kettle", "light switch", "slide cabinet"],
    "partial": ["microwave", "kettle", "light switch", "slide cabinet"],
    "mixed": ["microwave", "kettle", "bottom burner", "light switch"],
}
# Horizon des datasets Minari D4RL/kitchen/*-v2 : 450 pas d'après la documentation Minari (280 dans l'ancien
# D4RL v0). C'est une borne : l'environnement récupéré depuis le dataset Minari applique sa propre limite.
KITCHEN_HORIZON = 450


@dataclass(frozen=True)
class DatasetSpec:
    name: str                 # ex. "hopper-medium-replay-v2"
    domain: str               # halfcheetah | hopper | walker2d | kitchen
    quality: str              # medium | medium-replay | ...
    backend: str              # hdf5 | minari
    source: str               # URL HDF5 ou id Minari
    env_id: str               # id Gymnasium pour l'évaluation
    ref_min_score: float
    ref_max_score: float
    max_episode_len: int
    env_kwargs: dict = field(default_factory=dict, hash=False, compare=False)

    @property
    def filename(self) -> str:
        return self.source.rsplit("/", 1)[-1]

    def download_urls(self, sources=("official", "hf_mirror")) -> list[tuple[str, str]]:
        """[(source, URL)] à essayer dans l'ordre pour un dataset HDF5 (vide pour Minari)."""
        if self.backend != "hdf5":
            return []
        unknown = [s for s in sources if s not in D4RL_HDF5_SOURCES]
        if unknown:
            raise KeyError(f"sources inconnues {unknown} ; choix : {list(D4RL_HDF5_SOURCES)}")
        return [(s, D4RL_HDF5_SOURCES[s].format(filename=self.filename)) for s in sources]


def get_spec(name: str) -> DatasetSpec:
    """Analyse un nom de type '<domaine>-<qualité>-v2'."""
    parts = name.split("-")
    if len(parts) < 3 or not parts[-1].startswith("v"):
        raise ValueError(f"nom de dataset invalide : {name!r} (attendu '<domaine>-<qualité>-v2')")
    domain, quality = parts[0], "-".join(parts[1:-1])

    if domain in MUJOCO_ENVS:
        if quality not in MUJOCO_QUALITIES:
            raise ValueError(f"qualité inconnue {quality!r} pour {domain} ; choix : {MUJOCO_QUALITIES}")
        env_id, rmin, rmax = MUJOCO_ENVS[domain]
        fname = f"{domain}_{quality.replace('-', '_')}-v2.hdf5"
        return DatasetSpec(
            name=name, domain=domain, quality=quality, backend="hdf5",
            source=D4RL_HDF5_SOURCES["official"].format(filename=fname), env_id=env_id,
            ref_min_score=rmin, ref_max_score=rmax, max_episode_len=1000,
        )

    if domain == "kitchen":
        if quality not in KITCHEN_TASKS:
            raise ValueError(f"qualité inconnue {quality!r} pour kitchen ; choix : {list(KITCHEN_TASKS)}")
        return DatasetSpec(
            name=name, domain="kitchen", quality=quality, backend="minari",
            source=f"D4RL/kitchen/{quality}-v2", env_id="FrankaKitchen-v1",
            ref_min_score=0.0, ref_max_score=4.0, max_episode_len=KITCHEN_HORIZON,
            env_kwargs={"tasks_to_complete": KITCHEN_TASKS[quality]},
        )

    raise ValueError(f"domaine inconnu {domain!r} ; choix : {list(MUJOCO_ENVS) + ['kitchen']}")
