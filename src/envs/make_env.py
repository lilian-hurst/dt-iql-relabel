"""Création des environnements d'évaluation (Gymnasium) correspondant aux datasets D4RL."""
from __future__ import annotations

import warnings

import gymnasium as gym
import numpy as np

from src.data.registry import DatasetSpec, get_spec


class FlatKitchenObs(gym.ObservationWrapper):
    """FrankaKitchen-v1 renvoie un dict ; on ne garde que 'observation' (comme dans les datasets)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = env.observation_space["observation"]

    def observation(self, obs):
        return np.asarray(obs["observation"], dtype=np.float32)


def _make_kitchen(spec: DatasetSpec) -> gym.Env:
    try:
        import gymnasium_robotics
    except ImportError as e:
        raise ImportError("Franka Kitchen nécessite : pip install -r requirements-kitchen.txt") from e
    from src.envs.compat import patch_gymnasium_robotics

    patch_gymnasium_robotics()
    gym.register_envs(gymnasium_robotics)
    # Environnement d'évaluation enregistré dans le dataset Minari s'il est présent localement
    # (garantit les mêmes tâches / horizon que les données), sinon reconstruction depuis le registre.
    try:
        import minari

        env = minari.load_dataset(spec.source).recover_environment(eval_env=True)
    except Exception:  # noqa: BLE001  (dataset absent, minari absent, ...)
        env = gym.make(spec.env_id, max_episode_steps=spec.max_episode_len, **spec.env_kwargs)
    return FlatKitchenObs(env)


def make_env(spec: DatasetSpec | str, seed: int | None = None) -> gym.Env:
    if isinstance(spec, str):
        spec = get_spec(spec)

    if spec.domain == "kitchen":
        env = _make_kitchen(spec)
    else:
        # v4 (et non v5) volontairement : mêmes observations/récompenses que les envs v3
        # avec lesquels les datasets D4RL v2 ont été collectés (vérifié par le test de rejeu).
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*is out of date.*")
            env = gym.make(spec.env_id, max_episode_steps=spec.max_episode_len)

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env
