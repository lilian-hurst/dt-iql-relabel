"""Correctifs de compatibilité entre bibliothèques tierces.

gymnasium-robotics (<= 1.4.2) teste `joint_type in (mjtJoint.mjJNT_HINGE, mjtJoint.mjJNT_SLIDE)`
avec joint_type de type np.int32. Avec les versions récentes de MuJoCo (>= 3.3 environ), l'énumération
`mjtJoint` ne se compare plus via `in` à un np.int32 (alors que `==` fonctionne), et FrankaKitchen-v1
plante dès reset(). On remplace `mujoco.mjtJoint`, uniquement dans le module fautif, par des
entiers Python. Le correctif ne s'applique que si le bug est détecté.
"""
from __future__ import annotations

import types

import numpy as np

_PATCHED = False


class _MujocoProxy(types.ModuleType):
    """Délègue tout à `mujoco`, sauf `mjtJoint` (remplacé par des entiers)."""

    def __init__(self, mujoco_module):
        super().__init__("mujoco")
        self._mj = mujoco_module
        jt = mujoco_module.mjtJoint
        self.mjtJoint = types.SimpleNamespace(
            **{n: int(getattr(jt, n)) for n in ("mjJNT_FREE", "mjJNT_BALL", "mjJNT_SLIDE", "mjJNT_HINGE")}
        )

    def __getattr__(self, name):
        return getattr(self._mj, name)


def patch_gymnasium_robotics() -> bool:
    """Renvoie True si le correctif a été appliqué (une seule fois)."""
    global _PATCHED
    if _PATCHED:
        return True
    import mujoco

    hinge = mujoco.mjtJoint.mjJNT_HINGE
    if np.int32(int(hinge)) in (hinge,):
        return False  # versions compatibles : rien à faire
    from gymnasium_robotics.utils import mujoco_utils

    mujoco_utils.mujoco = _MujocoProxy(mujoco)
    _PATCHED = True
    return True
