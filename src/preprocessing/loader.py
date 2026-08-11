# src/preprocessing/loader.py

import subprocess
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple

import numpy as np
from tqdm import tqdm

from src.utils.paths import DATA

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 1. DÉCOUVERTE DES FICHIERS DICOM
# ─────────────────────────────────────────────────────────────

def find_dicom_files(dicoms_parent_folder: str, old_dicom_paths: list[str] | None = None) -> list[str]:
    """
    Parcourt l'arborescence et retourne tous les fichiers .dcm trouvés.

    Args:
        dicoms_parent_folder: Dossier racine contenant les .dcm.
        old_dicom_paths: Liste des DICOM déjà convertis à exclure (None = aucun).

    Returns:
        Liste des chemins complets vers chaque fichier .dcm.
    """
    old_dicom_paths = old_dicom_paths or []
    dcm_files = dcm_files = [p for p in Path(dicoms_parent_folder).rglob('*') if p.suffix.upper() == '.DCM']
    logger.info(f"{len(dcm_files)} fichiers DICOM trouvés dans {dicoms_parent_folder}")
    return [str(p) for p in dcm_files if not str(p) in old_dicom_paths]


def build_conversion_pairs(
    dicoms_parent_folder: str,
    niftis_parent_folder: str,
    old_dicom_paths: list[str] | None = None,
    save_paths: bool = True,
    paths_save_dir: str | Path = DATA,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Associe chaque fichier .dcm à son chemin .nii.gz de destination,
    en respectant l'arborescence des dossiers.

    Exemple :
        data/raw/patient_001/topo.dcm
        → data/processed/patient_001/topo.nii.gz

    Args:
        dicoms_parent_folder: Dossier racine DICOM.
        niftis_parent_folder: Dossier racine NIfTI de destination.
        old_dicom_paths: Liste des DICOM déjà convertis à exclure (None = aucun).
        save_paths:           Sauvegarder les chemins en .npy.
        paths_save_dir: Dossier de sauvegarde des .npy. DOIT rester synchronisé
                   avec DATA dans src.utils.paths (io.charger_dicom_paths
                   lit dicom_paths.npy depuis ce même dossier par défaut).

    Returns:
        Tuple (dicom_paths, nifti_paths) de shape (N,)
    """
    old_dicom_paths = old_dicom_paths or []
    dicom_files = find_dicom_files(dicoms_parent_folder, old_dicom_paths)

    dicom_root = Path(dicoms_parent_folder)
    nifti_root = Path(niftis_parent_folder)

    dicom_paths, nifti_paths = [], []

    for dcm_file in dicom_files:
        dcm_path = Path(dcm_file)

        # Reconstruction de l'arborescence côté NIfTI
        relative      = dcm_path.relative_to(dicom_root)
        nifti_file    = nifti_root / relative.parent / (dcm_path.stem + '.nii.gz')

        dicom_paths.append(str(dcm_path))
        nifti_paths.append(str(nifti_file))

    dicom_array = np.array(dicom_paths)
    nifti_array = np.array(nifti_paths)

    if save_paths:
        save_dir = Path(paths_save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / 'dicom_paths.npy', dicom_array)
        np.save(save_dir / 'nifti_paths.npy',  nifti_array)
        logger.info(f"Chemins sauvegardés dans {save_dir}")

    return dicom_array, nifti_array


# ─────────────────────────────────────────────────────────────
# 2. CONVERSION FICHIER PAR FICHIER
# ─────────────────────────────────────────────────────────────

def convert_dicom_to_nifti(
    input_file: str,
    output_file: str,
    silent: bool = True
) -> bool:
    """
    Convertit un fichier .dcm en .nii.gz via Plastimatch.

    Args:
        input_file:  Chemin vers le fichier .dcm.
        output_file: Chemin du .nii.gz à générer.
        silent:      Si False, affiche les messages début/fin.

    Returns:
        True si succès, False en cas d'échec — Plastimatch en erreur,
        timeout dépassé, ou toute autre exception (ex: Plastimatch non
        installé). Un échec sur un fichier ne doit jamais faire
        remonter d'exception : voir run_conversion_pipeline, qui
        s'appuie sur cette garantie pour continuer le lot même si
        un fichier échoue.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        logger.debug(f"Déjà converti, skip : {output_file}")
        return True

    command = [
        'plastimatch', 'convert',
        '--input',       input_file,
        '--output-type', 'short',
        '--output-img',  output_file
    ]

    try:
        if not silent:
            print(f"Conversion : {input_file}")

        subprocess.run(command, check=True, capture_output=True, timeout=60)

        if not silent:
            print(f"  → {output_file}")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Plastimatch échoué sur {input_file} :\n{e.stderr.decode()}")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout dépassé sur {input_file}")
        return False
    except Exception as e:
        # Filet de sécurité : Plastimatch absent (FileNotFoundError),
        # permissions, disque plein... N'importe quelle erreur inattendue
        # doit rester locale à CE fichier, jamais remonter jusqu'au
        # ThreadPoolExecutor (voir run_conversion_pipeline).
        logger.error(f"Erreur inattendue sur {input_file} : {type(e).__name__}: {e}")
        return False


# ─────────────────────────────────────────────────────────────
# 3. PIPELINE COMPLET AVEC PARALLÉLISME
# ─────────────────────────────────────────────────────────────

def run_conversion_pipeline(
    dicoms_parent_folder: str,
    niftis_parent_folder: str,
    old_dicom_paths: list[str] | None = None,
    n_workers: int = 4,
    save_paths: bool = True,
    paths_save_dir: str | Path = DATA,
) -> dict:
    """
    Pipeline complet : découverte + conversion parallèle de tous les .dcm.

    Args:
        dicoms_parent_folder: Dossier racine DICOM.
        niftis_parent_folder: Dossier racine NIfTI.
        old_dicom_paths: Liste des DICOM déjà convertis à exclure (None = aucun).
        n_workers:            Threads parallèles (4-8 selon CPU).
        save_paths:           Sauvegarder les chemins en .npy.
        paths_save_dir: Dossier de sauvegarde des .npy. DOIT rester synchronisé
                           avec DATA dans src.utils.paths (io.charger_dicom_paths
                           lit dicom_paths.npy depuis ce même dossier par défaut).

    Returns:
        dict avec 'total', 'success', 'failed', 'failed_paths'.
    """
    old_dicom_paths = old_dicom_paths or []
    dicom_paths, nifti_paths = build_conversion_pairs(
        dicoms_parent_folder, 
        niftis_parent_folder,
        old_dicom_paths,
        save_paths,
        paths_save_dir
    )
    

    n_total = len(dicom_paths)

    print(f"\n{'─'*50}")
    print(f"  {n_total} fichiers DICOM  •  {n_workers} workers")
    print(f"{'─'*50}\n")


    success, failed = [], []

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(convert_dicom_to_nifti, dcm, nii): (dcm, nii)
            for dcm, nii in zip(dicom_paths, nifti_paths)
        }

        with tqdm(total=n_total, desc="DICOM → NIfTI", unit="fichier") as pbar:
            for future in as_completed(futures):
                dcm, nii = futures[future]
                if future.result():
                    success.append(nii)
                else:
                    failed.append(dcm)
                pbar.update(1)

    stats = {
        'total':        n_total,
        'success':      len(success),
        'failed':       len(failed),
        'failed_paths': failed
    }

    print(f"\n{'─'*50}")
    print(f"  ✅ Succès : {stats['success']}/{n_total}")
    print(f"  ❌ Échecs : {stats['failed']}")
    for p in failed:
        print(f"     • {p}")
    print(f"{'─'*50}\n")

    return stats
