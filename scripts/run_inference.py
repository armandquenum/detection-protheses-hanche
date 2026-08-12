# scripts/run_inference.py
"""
Lance l'inférence d'un modèle (nnU-Net v1/v2 ou baseline SimpleITK) sur des
images NIfTI, et sort un résumé CSV avec une ligne par patient (colonnes :
présence de prothèse, latéralité, métadonnées DICOM disponibles).

Exemples :
    # Avec les valeurs par défaut (config.yaml : DATA_NIFTIS -> DATA_NNUNET_MASK)
    python scripts/run_inference.py

    # Liste explicite d'images, modèle nnU-Net v1, sur GPU
    python scripts/run_inference.py -i data/niftis/case1.nii.gz data/niftis/case2.nii.gz -m nnunet -d cuda

    # Baseline SimpleITK (pas de fold/dataset_id, ignorés pour ce modèle)
    python scripts/run_inference.py -m sitk -o data/processed/sitk_seg
"""

from pathlib import Path
import argparse

import pandas as pd

from src.utils.paths import DATA_NIFTIS, DATA_NNUNET_MASK
from src.annotation.pipeline_annotation import generer_annotations_candidates
from src.segmentation.predictor import predire_images_multiple, configurer_env_nnunet


def _est_nifti(path: Path) -> bool:
    """Vrai si `path` est un fichier NIfTI compressé (*.nii.gz)."""
    return path.name.lower().endswith(".nii.gz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Inference',
        description="Lance l'inférence d'un modèle sur des images Nifti",
        epilog='-' * 50,
    )

    parser.add_argument(
        '-i', '--input',
        type=str,
        nargs='*',
        # NOTE : nargs='*' attend une VALEUR ITÉRABLE en default (une liste),
        # pas un Path brut — sinon, quand l'option n'est pas fournie en CLI,
        # args.input reste un Path (non itérable) et la boucle plus bas plante.
        default=[str(DATA_NIFTIS)],
        help="Dossier racine contenant les images, ou liste explicite de "
             "fichiers .nii.gz séparés par des espaces "
             "(défaut : DATA_NIFTIS dans config.yaml)."
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        # Contrairement à --input, --output désigne UN SEUL dossier de
        # destination : pas de nargs='*' ici.
        default=str(DATA_NNUNET_MASK),
        help="Dossier racine dans lequel sont enregistrés les masques "
             "prédits et le CSV de résumé "
             "(défaut : DATA_NNUNET_MASK dans config.yaml)."
    )

    parser.add_argument(
        '-m', '--model',
        type=str,
        default="nnunetv2",
        choices=["nnunet", "nnunetv2", "sitk"],
        help="Modèle à utiliser pour l'inférence : 'nnunet' (v1), "
             "'nnunetv2' ou 'sitk' (baseline SimpleITK) (défaut : nnunetv2)."
    )

    parser.add_argument(
        '--dataset_id',
        type=int,
        default=1,
        help="ID numérique du dataset nnU-Net (ex : Dataset001_xxx -> 1). "
             "Utilisé uniquement pour les modèles 'nnunet'/'nnunetv2', "
             "ignoré pour 'sitk' (défaut : 1)."
    )

    parser.add_argument(
        '-f', '--fold',
        type=int,
        nargs='*',
        default=[0, 1, 2, 3, 4],
        metavar="FOLD",
        help="Liste des folds à utiliser pour les modèles nnU-Net, "
             "ex : -f 0 1 2. Ignoré pour 'sitk' (défaut : 0 1 2 3 4, soit "
             "les 5 folds de la validation croisée)."
    )

    parser.add_argument(
        '-d', '--device',
        type=str,
        default='cpu',
        choices=["cpu", "cuda"],
        help="Device utilisé pour l'inférence : 'cpu' ou 'cuda' (défaut : cpu)."
    )

    args = parser.parse_args()

    # ── Résolution des chemins d'entrée ─────────────────────────────────
    # args.input peut mélanger des dossiers (auquel cas on cherche tous les
    # .nii.gz dedans) et des fichiers .nii.gz donnés directement.
    niftis_path = []
    for i in args.input:
        path = Path(i)
        if not path.exists():
            continue
        if path.is_dir():
            niftis_path.extend(str(p) for p in path.rglob('*') if _est_nifti(p))
        elif path.is_file() and _est_nifti(path):
            # Bug corrigé : `path.is_file` sans parenthèses est toujours
            # "truthy" (c'est une référence de méthode, pas un appel), donc
            # cette branche était toujours prise dès que ce n'était pas un
            # dossier — même pour un fichier qui n'existait pas ou n'était
            # pas un .nii.gz.
            niftis_path.append(str(path))

    # ── Inférence ────────────────────────────────────────────────────────
    
    columns_to_keep = [
        "prothese",
        "lateralite",
        "jour_d_examen",
        "date_de_naissance",
        "requested_procedure_description",
        "scheduled_procedure_step_description",
    ]
    segmentations = {}
    if args.model in ("nnunet", "nnunetv2"):
        configurer_env_nnunet(args.model)
        segmentations = predire_images_multiple(
            niftis_path,
            masks_folder=args.output,
            dataset_id=args.dataset_id,
            fold=list(args.fold),
            device=args.device,
            mode='infer'
        )
    else:
        segmentations = generer_annotations_candidates(
            nifti_paths=niftis_path,
            output_dir=args.output,
            mode='annotation'
        )

    # ── Construction du DataFrame ───────────────────────────────────────
    # NOTE IMPORTANTE : on ne peut pas faire
    #   pd.DataFrame.from_dict(segmentations, orient="index", columns=columns_to_keep)
    # directement. Avec orient="index", le paramètre `columns` ne sélectionne
    # PAS les clés demandées : il renomme POSITIONNELLEMENT les colonnes que
    # pandas a déjà déduites de l'union des clés rencontrées. Si un patient
    # n'a pas exactement les mêmes clés que les autres (ex : DICOM introuvable
    # -> pas de métadonnées ajoutées pour lui), l'ordre des colonnes peut
    # varier et le résultat se retrouve mal étiqueté silencieusement.
    # On construit donc explicitement, pour chaque patient, un sous-dict
    # limité aux colonnes voulues (avec None si la clé est absente).
    lignes = {
        patient_id: {col: data.get(col) for col in columns_to_keep}
        for patient_id, data in segmentations.items()
    }
    df = pd.DataFrame.from_dict(lignes, orient="index", columns=columns_to_keep)
    df.index.name = "patient_id"
    df = df.reset_index()


    df["lateralite"] = pd.Categorical(
        df["lateralite"],
        categories=['pas_de_prothese', 'gauche', 'droite', 'bilatere', 'profil'],
    )
    df["prothese"] = pd.Categorical(
        df["prothese"],
        categories=['non', 'oui'],
    )

    
    csv_path = Path(args.output) / 'segmentations.csv'
    df.to_csv(csv_path, index=False)
    print(f"Résumé sauvegardé : {csv_path}")