"""Projet DT + relabellisation des return-to-go par IQL."""
import os

# Apple Silicon (backend MPS) : une opération PyTorch non implémentée sur le GPU retombe sur le CPU
# au lieu de lever une erreur. Doit être défini avant le premier import de torch, d'où sa place ici.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
