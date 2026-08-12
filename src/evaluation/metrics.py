# src/evaluation/metrics.py

import logging
import numpy as np
import SimpleITK as sitk
from scipy.spatial.distance import directed_hausdorff

logger = logging.getLogger(__name__)


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Coefficient Dice — mesure le chevauchement entre prédiction et vérité terrain.

    Dice = 2 * |P ∩ G| / (|P| + |G|)
    1.0 = parfait, 0.0 = aucun chevauchement.
    Si pred et gt sont tous les deux vides → retourne 1.0 (cas vrai négatif).
    """
    pred = pred.astype(bool).ravel()
    gt   = gt.astype(bool).ravel()

    if not pred.any() and not gt.any():
        return 1.0   # Vrai négatif parfait

    intersection = (pred & gt).sum()
    denom        = pred.sum() + gt.sum()
    return float(2 * intersection / denom) if denom > 0 else 0.0


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Intersection over Union (Jaccard).
    IoU = |P ∩ G| / |P ∪ G|
    """
    pred = pred.astype(bool).ravel()
    gt   = gt.astype(bool).ravel()

    if not pred.any() and not gt.any():
        return 1.0

    intersection = (pred & gt).sum()
    union        = (pred | gt).sum()
    return float(intersection / union) if union > 0 else 0.0


def precision_recall(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """
    Précision et rappel au niveau pixel.

    Précision = TP / (TP + FP)  — parmi les pixels prédits positifs, combien sont corrects ?
    Rappel    = TP / (TP + FN)  — parmi les vrais positifs, combien sont détectés ?
    """
    pred = pred.astype(bool).ravel()
    gt   = gt.astype(bool).ravel()

    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall    = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    return precision, recall


def hausdorff_distance(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: tuple[float, ...] = (1.0, 1.0),
    percentile: float = 95.0,
) -> float:
    """
    Distance de Hausdorff entre les contours de la prédiction et la vérité.
    Mesure la précision des bords — importante pour les prothèses.

    Args:
        pred:       Masque prédit (2D ou 3D).
        gt:         Masque de référence.
        spacing:    Espacement physique entre voxels (pour distance en mm).
        percentile: Hausdorff au percentile X (95HD recommandé en médical).

    Returns:
        Distance en mm (float). np.inf si un masque est vide.
    """
    def contour_coords(mask, sp):
        """Extraire les coordonnées physiques des pixels de contour."""
        mask = mask.astype(np.uint8)
        # Contour = pixels qui diffèrent de leurs voisins
        from scipy.ndimage import binary_erosion
        eroded  = binary_erosion(mask)
        contour = mask ^ eroded
        coords  = np.argwhere(contour)
        # np.argwhere(mask) suit l'ordre du tableau numpy (Z, Y, X) — issu
        # de sitk.GetArrayFromImage — alors que sp (GetSpacing()) est en
        # ordre SimpleITK (X, Y, Z). On inverse sp pour aligner les deux,
        # sinon l'axe X hérite du spacing Z (souvent 1.0 pour les images
        # 2D encapsulées via JoinSeries) et les distances en mm sont fausses.
        sp_array_order = np.array(sp)[::-1]
        return coords * sp_array_order

    coords_pred = contour_coords(pred, spacing)
    coords_gt   = contour_coords(gt, spacing)

    if len(coords_pred) == 0 or len(coords_gt) == 0:
        return np.inf

    # Hausdorff directionnel dans les deux sens
    d_p2g = directed_hausdorff(coords_pred, coords_gt)[0]
    d_g2p = directed_hausdorff(coords_gt, coords_pred)[0]

    if percentile == 100.0:
        return float(max(d_p2g, d_g2p))

    # Hausdorff au percentile (95HD)
    from scipy.spatial import cKDTree
    tree_gt   = cKDTree(coords_gt)
    tree_pred = cKDTree(coords_pred)

    dist_p2g = tree_gt.query(coords_pred)[0]
    dist_g2p = tree_pred.query(coords_gt)[0]

    return float(max(
        np.percentile(dist_p2g, percentile),
        np.percentile(dist_g2p, percentile)
    ))


def volume_error(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: tuple[float, ...] = (1.0, 1.0, 1.0),
) -> dict[str, float]:
    """
    Erreur volumétrique entre prédiction et vérité.

    Returns:
        dict avec 'vol_pred_mm3', 'vol_gt_mm3', 'erreur_absolue_mm3', 'erreur_relative_pct'.
    """
    voxel_vol = float(np.prod(spacing))
    vol_pred  = float(pred.astype(bool).sum()) * voxel_vol
    vol_gt    = float(gt.astype(bool).sum())   * voxel_vol
    err_abs   = abs(vol_pred - vol_gt)
    err_rel   = (err_abs / vol_gt * 100) if vol_gt > 0 else np.inf

    return {
        "vol_pred_mm3":        vol_pred,
        "vol_gt_mm3":          vol_gt,
        "erreur_absolue_mm3":  err_abs,
        "erreur_relative_pct": err_rel,
    }


def detection_correcte(pred: np.ndarray, gt: np.ndarray) -> dict[str, bool]:
    """
    Évaluation au niveau patient (détection binaire).
    Indépendant de la qualité de la segmentation.

    Returns:
        dict avec 'vrai_positif', 'vrai_negatif', 'faux_positif', 'faux_negatif'.
    """
    pred_pos = pred.astype(bool).any()
    gt_pos   = gt.astype(bool).any()
    return {
        "vrai_positif":  bool(pred_pos and gt_pos),
        "vrai_negatif":  bool(not pred_pos and not gt_pos),
        "faux_positif":  bool(pred_pos and not gt_pos),
        "faux_negatif":  bool(not pred_pos and gt_pos),
    }
def detection_lateralite(pred_lateralite: int, gt_lateralite: int) -> dict[str, np.ndarray]:
    """
    Évaluation au niveau patient de la latéralité — construit une matrice
    de confusion "unitaire" (une seule case à True) pour cette paire
    (pred, gt). Voir evaluator._afficher_resume pour l'agrégation par
    sommation de ces matrices sur tout le dataset.

    Les valeurs -1 (orientation indéterminée, voir io.calculer_lateralite)
    sont exclues du calcul : en Python, confusion_matrix[-1, ...] indexe
    silencieusement la DERNIÈRE ligne (index 4) plutôt que de lever une
    erreur — sans ce garde-fou, un cas indéterminé (-1) se confondrait
    avec la valeur 4 (vue de profil, latéralité non calculable), deux
    significations distinctes. Le dataset ne devrait normalement jamais
    produire -1 à ce stade (déjà exclu à la validation des annotations,
    voir pipeline_annotation._sauvegarder_masque_corrige) — ce garde-fou
    protège uniquement contre une régression amont.

    Returns:
        dict avec 'confusion_matrix' (5x5 bool). Entièrement à False
        (ne contribue à rien lors de la sommation) si gt ou pred vaut -1.
    """
    confusion_matrix = np.full((5, 5), False)

    if gt_lateralite == -1 or pred_lateralite == -1:
        logger.warning(
            f"Latéralité indéterminée (-1) rencontrée en évaluation "
            f"(gt={gt_lateralite}, pred={pred_lateralite}) — exclue de la "
            f"matrice de confusion (ne devrait normalement pas arriver ici)."
        )
        return {"confusion_matrix": confusion_matrix}

    confusion_matrix[gt_lateralite, pred_lateralite] = True
    return {
        "confusion_matrix":  confusion_matrix,
    }

def toutes_les_metriques(
    pred_info: dict,
    gt_info: dict,
) -> dict:
    """
    Calcule toutes les métriques pour une paire (prédiction, vérité terrain).

    Args:
        pred_info: dict avec au minimum 'mask' (chemin NIfTI du masque
                   prédit) et 'lateralite' (int, voir calculer_lateralite).
        gt_info:   dict avec la même structure, pour la vérité terrain.

    Returns:
        Dictionnaire complet de métriques.
    """
    pred_sitk = sitk.ReadImage(pred_info["mask"])
    gt_sitk = sitk.ReadImage(gt_info["mask"])
    pred    = sitk.GetArrayFromImage(pred_sitk).astype(np.uint8)
    gt      = sitk.GetArrayFromImage(gt_sitk).astype(np.uint8)
    spacing = pred_sitk.GetSpacing()   # (sx, sy, sz) en mm

    dice         = dice_score(pred, gt)
    iou          = iou_score(pred, gt)
    prec, recall = precision_recall(pred, gt)
    f1           = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
    hd95         = hausdorff_distance(pred, gt, spacing=spacing, percentile=95.0)
    vol          = volume_error(pred, gt, spacing=spacing)
    det          = detection_correcte(pred, gt)
    
    det_lat      = detection_lateralite(pred_info["lateralite"], gt_info["lateralite"])

    return {
        # Métriques de segmentation
        "dice":                 round(dice,  4),
        "iou":                  round(iou,   4),
        "precision":            round(prec,  4),
        "recall":               round(recall,4),
        "f1":                   round(f1,    4),
        "hausdorff_95_mm":      round(hd95,  2) if not np.isinf(hd95) else None,
        # Métriques volumétriques
        **{k: round(v, 2) for k, v in vol.items()},
        # Métriques de détection (niveau patient)
        **det,
        **det_lat,
    }