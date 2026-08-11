# ─────────────────────────────────────────────
# src/utils/paths.py
#
# Point d'entrée unique pour tous les chemins du projet.
# Ce module est importé par presque tout le reste du code (io.py,
# visualization.py, scripts/, notebooks/) — toute modification ici
# a un impact global. Les constantes ROOT/CONFIG/DATA_RAW/... sont
# calculées UNE FOIS à l'import (voir bas de fichier), pas à la demande.
# ─────────────────────────────────────────────

import yaml
from pathlib import Path


def get_project_root() -> Path:
    """
    Localise la racine du projet en remontant l'arborescence depuis ce
    fichier jusqu'à trouver un dossier contenant pyproject.toml.

    Returns:
        Chemin absolu vers la racine du projet.

    Raises:
        FileNotFoundError: si aucun pyproject.toml n'est trouvé en
            remontant l'arborescence (ex: fichier déplacé hors du repo).
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Racine introuvable")


def load_config() -> dict:
    """
    Charge config.yaml depuis la racine du projet.

    Returns:
        Contenu de config.yaml sous forme de dictionnaire
        (voir la clé "paths" pour les chemins utilisés ci-dessous).

    Raises:
        FileNotFoundError: si config.yaml est absent de la racine.
    """
    config_path = get_project_root() / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


ROOT   = get_project_root()
CONFIG = load_config()

# Chemins depuis config.yaml si définis, sinon valeurs par défaut
DATA             = ROOT / CONFIG["paths"].get("data",       "data")
DATA_RAW         = ROOT / CONFIG["paths"].get("data_raw",       "data/raw")
DATA_PROCESSED   = ROOT / CONFIG["paths"].get("data_processed", "data/processed")
DATA_NIFTIS      = ROOT / CONFIG["paths"].get("data_niftis",    "data/processed/niftis")
ANNOTATIONS      = ROOT / CONFIG["paths"].get("annotations",    "data/annotations")
RESULTS          = ROOT / CONFIG["paths"].get("results",        "reports/figures")
MODELS           = ROOT / CONFIG["paths"].get("models",         "models")
NNUNET           = MODELS/ CONFIG["paths"].get("nnunet",         "nnunet")
NNUNETV2         = MODELS/ CONFIG["paths"].get("nnunetv2",         "nnunetv2")
NNUNET_DATASET   = ANNOTATIONS/ CONFIG["paths"].get("nnunet_dataset",   "NNUNET/nnUNet_raw/Dataset001_Protheses")
NNUNETV2_DATASET = ANNOTATIONS/ CONFIG["paths"].get("nnunetv2_dataset",   "NNUNET_V2/nnUNet_raw/Dataset001_Protheses")
FIGURES          = ROOT / CONFIG["paths"].get("figures",         "figures")
DATA_NNUNET_MASK = ROOT / CONFIG["paths"].get("data_nnunet_mask",       "data/nnunet_mask")