# src/evaluation/evaluator.py

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.annotation.pipeline_annotation import generer_annotations_candidates
from src.evaluation.metrics import toutes_les_metriques
from src.segmentation.predictor import predire_images_multiple, configurer_env_nnunet
from src.utils.io import charger_json, dossier_predictions
from src.utils.paths import RESULTS

LATERALITE = ["no_protesis", "left", "right", "both", "sagital"]

logger = logging.getLogger(__name__)


def evaluer_dataset_test(
    dataset_path: str,
    dataset_id: int = 1,
    config: str = "2d",
    fold: list[str] | list[int] | str | int = [0, 1, 2, 3, 4],
    device: str = "cpu",
    lancer_inference: bool = True,
    model: str = 'nnunet',
) -> pd.DataFrame:
    """
    Pipeline complet d'évaluation sur le jeu de test.

    1. Lance les prédictions (nnU-Net ou SimpleITK selon `model`), ou
       recharge les prédictions déjà calculées si lancer_inference=False
       (voir dossier_predictions pour le mapping modèle -> dossier).
    2. Compare chaque prédiction au masque de référence (gt.json).
    3. Calcule toutes les métriques (voir metrics.toutes_les_metriques).
    4. Retourne un DataFrame + sauvegarde CSV + affiche le résumé.

    Args:
        dataset_path:       Chemin vers le Dataset sur lequel on fait le test.
        dataset_id:         ID dataset nnU-Net (model='nnunet'/'nnunetv2').
        config:             Configuration nnU-Net.
        fold:               Fold du modèle nnU-Net.
        device:             'cpu', 'cuda', 'mps'.
        lancer_inference:   False si les prédictions existent déjà — elles
                            sont alors rechargées depuis
                            dossier_predictions(model, dataset_path) plutôt
                            que de relancer l'inférence.
        model:              Modèle à évaluer : 'nnunet', 'nnunetv2' ou 'sitk'.

    Returns:
        DataFrame avec une ligne par patient et toutes les métriques.
        DataFrame vide si `model` n'est pas reconnu.
    """
    version = 2 if model == 'nnunetv2' else 1

    match model:
        case 'nnunet' | 'nnunetv2':
            configurer_env_nnunet(model)
            if lancer_inference:
                print(f"Lancement des prédictions {model}...")
                pred_json = predire_images_multiple(
                    dataset_id=dataset_id, config=config,
                    fold=fold, device=device,
                    dataset_path=dataset_path,
                    mode='eval',
                    version=version,
                )
            else:
                pred_json = charger_json(
                    dossier_predictions(model, dataset_path, version) / 'segmentations.json'
                )

        case 'sitk':
            if lancer_inference:
                print("Lancement des prédictions SimpleITK...")
                pred_json = generer_annotations_candidates(
                    [], '', dataset_path=dataset_path, mode='eval'
                )
            else:
                pred_json = charger_json(
                    dossier_predictions('sitk', dataset_path) / 'segmentations.json'
                )

        case _:
            logger.warning(f"model '{model}' non reconnu, choisir entre 'nnunet'/'nnunetv2'/'sitk'")
            return pd.DataFrame()

    gt_json = charger_json(Path(dataset_path) / 'gt.json')

    resultats = []
    for case_id in tqdm(gt_json.keys(), desc="Évaluation", unit="patient"):
        gt_info   = gt_json.get(case_id)
        pred_info = pred_json.get(case_id)

        # Un cas manquant est écarté individuellement — ne doit jamais
        # faire perdre les résultats déjà calculés pour les autres cas.
        if pred_info is None or gt_info is None:
            logger.warning(f"Prédiction ou GT manquant pour {case_id}")
            resultats.append({"case_id": case_id, "erreur": "pred_ou_gt_manquant"})
            continue

        try:
            pred_mask = pred_info.get('mask')
            gt_mask   = gt_info.get('mask')
            if not pred_mask or not gt_mask:
                raise ValueError("clé 'mask' absente de pred_info ou gt_info")

            pred_path = Path(pred_mask)
            gt_path   = Path(gt_mask)

            if not pred_path.exists():
                logger.warning(f"Prédiction manquante : {pred_path}")
                resultats.append({"case_id": case_id, "erreur": "prediction_manquante"})
                continue
            if not gt_path.exists():
                logger.warning(f"Label manquant : {gt_path}")
                resultats.append({"case_id": case_id, "erreur": "label_manquant"})
                continue

            metriques = toutes_les_metriques(pred_info, gt_info)
            metriques["case_id"] = case_id
            resultats.append(metriques)

        except Exception as e:
            logger.error(f"Erreur sur {case_id} : {e}")
            resultats.append({"case_id": case_id, "erreur": str(e)})

    df = pd.DataFrame(resultats)

    # ── Sauvegarde ──────────────────────────────────────────────────────
    RESULTS.mkdir(parents=True, exist_ok=True)
    # Nom de fichier propre à model+dataset : évite d'écraser les résultats
    # d'un autre modèle ou d'un autre dataset évalué juste avant.
    csv_path = RESULTS / f"evaluation_{model}_{Path(dataset_path).name}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Résultats sauvegardés : {csv_path}")

    # ── Résumé ──────────────────────────────────────────────────────────
    _afficher_resume(df)
    return df


def _afficher_resume(df: pd.DataFrame) -> None:
    """Affiche un tableau récapitulatif des métriques."""
    cols_seg = ["dice", "iou", "precision", "recall", "hausdorff_95_mm"]
    cols_det = ["vrai_positif", "vrai_negatif", "faux_positif", "faux_negatif"]

    if "erreur" in df.columns:
        df_valides = df[df["erreur"].isna()]        # garde les lignes sans erreur
    else:
        df_valides = df.copy()
    n_total = len(df_valides)

    print(f"\n{'═'*55}")
    print(f"  RÉSULTATS — {n_total} patients")
    print(f"{'═'*55}")

    # Métriques de segmentation (sur les vrais positifs uniquement)
    df_tp = df_valides[df_valides.get("vrai_positif", False) == True]
    if len(df_tp) > 0:
        print(f"\n  Segmentation (sur {len(df_tp)} vrais positifs) :")
        for col in cols_seg:
            if col in df_tp.columns:
                vals = df_tp[col].dropna()
                if len(vals):
                    print(f"    {col:25s} {vals.mean():.4f} ± {vals.std():.4f}")

    # Métriques de détection
    if all(c in df_valides.columns for c in cols_det):
        tp = df_valides["vrai_positif"].sum()
        tn = df_valides["vrai_negatif"].sum()
        fp = df_valides["faux_positif"].sum()
        fn = df_valides["faux_negatif"].sum()
        n  = tp + tn + fp + fn

        sensibilite = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificite = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy    = (tp + tn) / n if n > 0 else 0

        print(f"\n  Détection (niveau patient) :")
        print(f"    {'Vrais positifs':25s} {tp}/{tp+fn}")
        print(f"    {'Vrais négatifs':25s} {tn}/{tn+fp}")
        print(f"    {'Faux positifs':25s} {fp}")
        print(f"    {'Faux négatifs':25s} {fn}")
        print(f"    {'Sensibilité (rappel)':25s} {sensibilite:.4f}")
        print(f"    {'Spécificité':25s} {specificite:.4f}")
        print(f"    {'Accuracy':25s} {accuracy:.4f}")

    # Métriques de latéralité
    if "confusion_matrix" in df_valides.columns:
        confusion_matrix = df_valides["confusion_matrix"].apply(lambda x: x.astype(np.int8)).sum()

        print(f"\n  Matrice de confusion de la latéralité (niveau patient) :")
        print(pd.DataFrame(confusion_matrix, columns=LATERALITE, index=LATERALITE))

    print(f"{'═'*55}\n")