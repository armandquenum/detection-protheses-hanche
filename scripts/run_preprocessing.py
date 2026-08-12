# scripts/run_preprocessing.py
"""
Convertit un dossier de fichiers DICOM en un dossier de fichiers NIfTI
(via Plastimatch, voir src.preprocessing.loader.run_conversion_pipeline).

Exemples :
    # Avec les dossiers par défaut (config.yaml : data_raw / data_niftis)
    python scripts/run_preprocessing.py

    # Dossiers explicites, 8 threads
    python scripts/run_preprocessing.py -d data/raw -n data/processed/niftis -w 8

    # Reprise après interruption : exclut les DICOM déjà convertis lors
    # d'une exécution précédente (fichier dicom_paths.npy généré par
    # cette exécution précédente, voir loader.build_conversion_pairs)
    python scripts/run_preprocessing.py -e data/dicom_paths.npy
"""

import argparse
from pathlib import Path

import numpy as np

from src.utils.paths import DATA_RAW, DATA_NIFTIS
from src.preprocessing.loader import run_conversion_pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Preprocessing',
        description='Convertit un dossier contenant des DICOM en un dossier contenant des NIfTI',
        epilog='-' * 50,
    )

    parser.add_argument(
        '-d', '--dicoms_parent_folder',
        type=str,
        default=str(DATA_RAW),
        help="Dossier racine contenant les fichiers DICOM à convertir (défaut : DATA_RAW dans config.yaml)."
    )
    parser.add_argument(
        '-n', '--niftis_parent_folder',
        type=str,
        default=str(DATA_NIFTIS),
        help="Dossier racine de destination des fichiers NIfTI générés (défaut : DATA_NIFTIS dans config.yaml)."
    )
    parser.add_argument(
        '-e', '--exlcuded_dicoms_path',
        type=str,
        default=None,
        help=(
            "Chemin vers un fichier .npy contenant les chemins DICOM déjà "
            "convertis à exclure de cette exécution (reprise après "
            "interruption) — typiquement le dicom_paths.npy sauvegardé "
            "par une exécution précédente. Omis = aucune exclusion."
        )
    )
    parser.add_argument(
        '-w', '--n_workers',
        type=int,
        default=4,
        help="Nombre de threads parallèles pour la conversion (4-8 selon le CPU, défaut : 4)."
    )

    args = parser.parse_args()

    # exlcuded_dicoms_path est fourni comme un CHEMIN vers un .npy (pas une
    # liste directement en ligne de commande) — on le charge ici, pour
    # que run_conversion_pipeline reçoive bien list[str] | None comme
    # attendu par sa signature.
    exlcuded_dicoms_path = None
    if args.exlcuded_dicoms_path:
        chemin_exclusion = Path(args.exlcuded_dicoms_path)
        if not chemin_exclusion.exists():
            parser.error(f"--exlcuded_dicoms_path : fichier introuvable : {chemin_exclusion}")
        exlcuded_dicoms_path = np.load(chemin_exclusion).tolist()

    run_conversion_pipeline(
        dicoms_parent_folder=args.dicoms_parent_folder,
        niftis_parent_folder=args.niftis_parent_folder,
        old_dicom_paths=exlcuded_dicoms_path,
        n_workers=args.n_workers,
    )