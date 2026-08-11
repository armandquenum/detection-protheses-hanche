# ─────────────────────────────────────────────
# src/detection/constants.py
# ─────────────────────────────────────────────

# Critères morphologiques pour une prothèse de hanche (topogramme 2D)
SURFACE_MIN_MM2 = 1500    # Surface minimale d'une prothèse candidate
SURFACE_MAX_MM2 = 5000   # Surface maximale (évite les grands artefacts)
ELONG_MIN       = 3.0    # Élongation minimale (forme allongée)
ELONG_MAX       = 7.0    # Élongation maximale
LARGEUR_MIN_PX  = 4      # Largeur minimale moyenne (run-length encoding)