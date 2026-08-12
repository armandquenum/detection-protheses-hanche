"""
reconstruction_methods.py

Reconstruction 3D depuis des projections 2D multi-angles : rétroprojection,
combinaison en volume 3D (Visual Hull), 4 méthodes de reconstruction
comparées, et évaluation sur le dataset TotalSegmentator (image + GT 3D).

Séparation avec roi_projection.py (voir ARCHITECTURE.md §2) : ce module
couvre tout ce qui va du masque 2D vers le volume 3D (l'inverse de la
projection) — roi_projection.py ne couvre que le sens 3D -> 2D et sa
préparation (pad_roi_margin, génération de dataset d'entraînement).

Fonctions principales :
  - retroprojeter_masque : pendant exact de projeter, sens inverse.
  - backproject           : dispatcher vers les 3 backprojections "par
                             angle" (tube_plein/argmax_filtre/seuil_hu) —
                             space_carving a une granularité différente
                             (tous les angles à la fois), gérée séparément
                             dans reconstruct_3d.
  - reconstruct_3d        : reconstruction "de production" — choisit UNE
                             méthode (paramètre `method`, les 4 mêmes que
                             comparer_methodes) et combine tous les angles.
                             C'est ici qu'on branche la méthode retenue une
                             fois comparer_methodes exécutée.
  - comparer_methodes     : compare les 4 méthodes sur UN cas (délègue à
                             reconstruct_3d pour chacune, ne duplique plus
                             la logique de combinaison par méthode).
  - evaluate_reconstruction_on_dataset : évalue UNE méthode (reconstruct_3d)
                             sur un dataset entier.

4 méthodes de reconstruction (paramètre `method` de reconstruct_3d /
backproject, valeurs comparées par comparer_methodes) :
  1. tube_plein     : Visual Hull classique (extrusion uniforme) — la
                       seule des 4 qui est purement géométrique, aucun
                       besoin de l'image CT réelle (voir backproject).
  2. argmax_filtre  : extrusion restreinte aux indices [max-alpha, max+alpha]
  3. seuil_hu       : extrusion restreinte aux indices dont le HU
                       correspond à un métal (seuil physique)
  4. space_carving  : sculpture itérative par cohérence photométrique
                       multi-vues (au lieu d'un filtre par colonne isolé)

Ce module ne redéfinit ni le noyau géométrique de rotation ni les
métriques :
  - AXIS_NAMES / _rotate_plane_stack sont importés depuis roi_projection.py
    (seule source de ce noyau — projeter en dépend directement, d'où son
    maintien là-bas malgré la séparation projection/reconstruction) ;
  - dice_score / iou_score / hausdorff_distance sont importés depuis
    src.evaluation.metrics (seule source des métriques — voir
    ARCHITECTURE.md §6). hausdorff_distance y est restreinte au CONTOUR
    du masque : une version locale calculée sur tous les voxels positifs
    gonflerait artificiellement la distance des méthodes qui sur-
    segmentent (tube_plein en particulier), biaisant la comparaison
    elle-même.
"""

from __future__ import annotations
from typing import Literal, TYPE_CHECKING

import numpy as np
import SimpleITK as sitk

from roi_projection import AXIS_NAMES, _rotate_plane_stack
from src.evaluation.metrics import dice_score, iou_score, hausdorff_distance

if TYPE_CHECKING:
    import pandas


# --------------------------------------------------------------------------- #
# Rétroprojection : pendant exact de projeter, sens inverse
# --------------------------------------------------------------------------- #

def retroprojeter_masque(
    mask_2d: np.ndarray,
    theta_deg: float,
    volume_shape_zyx: tuple[int, int, int],
    axe_invariant: str = "Z",
    order: int = 0,
) -> np.ndarray:
    """
    Rétroprojette un masque 2D (issu de roi_projection.projeter à
    theta_deg) en un volume binaire 3D (Z, Y, X), dans le repère original
    du volume.

    Principe : le masque 2D est extrudé (répété) sur toute la profondeur
    de l'axe qui avait été agrégé — chaque pixel positif du masque devient
    une "colonne" pleine dans le plan tourné, représentant l'incertitude
    sur la position en profondeur. Cette colonne est ensuite ramenée dans
    le repère original par la même transformation géométrique que
    projeter (_rotate_plane_stack), appliquée à -theta_deg — c'est
    l'opération inverse exacte, pas une approximation.

    volume_shape_zyx : shape (Z, Y, X) du volume de référence dans lequel
                        toutes les rétroprojections (angles différents)
                        doivent être comparées/combinées — typiquement la
                        shape de la ROI (paddée si pad_roi_margin a été
                        utilisée en amont de la projection).
    """
    if axe_invariant not in AXIS_NAMES:
        raise ValueError(f"axe_invariant doit être 'Z', 'Y' ou 'X', reçu '{axe_invariant}'.")
    inv_idx = AXIS_NAMES.index(axe_invariant)

    dummy = np.empty(volume_shape_zyx)
    N_inv, A, B = np.moveaxis(dummy, inv_idx, 0).shape

    if mask_2d.shape != (N_inv, B):
        raise ValueError(
            f"mask_2d.shape={mask_2d.shape} incompatible avec la shape attendue "
            f"({N_inv}, {B}) pour axe_invariant='{axe_invariant}' et volume_shape_zyx="
            f"{volume_shape_zyx}. Vérifie que le même axe_invariant et la même ROI "
            f"(éventuellement paddée) ont été utilisés pour la projection."
        )

    extruded = np.broadcast_to(mask_2d[:, np.newaxis, :], (N_inv, A, B)).astype(np.float32)
    rotated_back = _rotate_plane_stack(extruded, -theta_deg, order=order)
    rotated_back = np.nan_to_num(rotated_back, nan=0.0)

    volume_moved = (rotated_back > 0.5).astype(np.uint8)
    return np.moveaxis(volume_moved, 0, inv_idx)


# --------------------------------------------------------------------------- #
# Méthode 2 — Argmax filtré : extrusion restreinte aux indices proches du max
# --------------------------------------------------------------------------- #

def _to_moved(image_3d: sitk.Image, axe_invariant: str) -> tuple[np.ndarray, int]:
    inv_idx = AXIS_NAMES.index(axe_invariant)
    array = sitk.GetArrayFromImage(image_3d).astype(np.float32)
    return np.moveaxis(array, inv_idx, 0), inv_idx


def backproject_argmax_filtre(
    mask_2d: np.ndarray, image_3d: sitk.Image, theta_deg: float,
    axe_invariant: str = "Z", alpha: float = 100.0,
) -> np.ndarray:
    """
    Au lieu d'extruder tout le tube, ne remplit que les indices (le long
    de l'axe agrégé) dont l'intensité HU est dans [max-alpha, max+alpha]
    de la coupe correspondante — utilise l'image d'origine, pas seulement
    le masque.
    """
    array_moved, inv_idx = _to_moved(image_3d, axe_invariant)
    N_inv, A, B = array_moved.shape

    # Rotation de l'IMAGE (pas du masque) à theta_deg pour connaître les
    # intensités le long de chaque rayon de projection
    rotated_img = _rotate_plane_stack(array_moved, theta_deg, order=1)
    # rotated_img[n, a, b] = intensité HU au point (n, a, b) dans le
    # repère tourné, où b correspond à l'axe D de la projection

    volume_filtre = np.zeros((N_inv, A, B), dtype=np.float32)

    for n in range(N_inv):
        for b in range(B):
            if mask_2d[n, b] <= 0.5:
                continue  # masque nul à cette position -> rien à remplir
            colonne = rotated_img[n, :, b]
            if np.all(np.isnan(colonne)):
                continue
            max_val = np.nanmax(colonne)
            indices_valides = np.where(
                (colonne >= max_val - alpha) & (colonne <= max_val + alpha)
            )[0]
            volume_filtre[n, indices_valides, b] = 1.0

    rotated_back = _rotate_plane_stack(volume_filtre, -theta_deg, order=0)
    rotated_back = np.nan_to_num(rotated_back, nan=0.0)
    volume_moved = (rotated_back > 0.5).astype(np.uint8)
    return np.moveaxis(volume_moved, 0, inv_idx)


# --------------------------------------------------------------------------- #
# Méthode 3 — Seuil HU physique (métal de prothèse)
# --------------------------------------------------------------------------- #

def backproject_seuil_hu(
    mask_2d: np.ndarray, image_3d: sitk.Image, theta_deg: float,
    axe_invariant: str = "Z", seuil_hu_metal: float = 1500.0,
) -> np.ndarray:
    """
    Ne remplit que les indices dont le HU dépasse un seuil physique
    caractéristique du métal (typiquement > 1000-1500 HU pour du
    titane/chrome-cobalt en CT clinique).
    """
    array_moved, inv_idx = _to_moved(image_3d, axe_invariant)
    N_inv, A, B = array_moved.shape

    rotated_img = _rotate_plane_stack(array_moved, theta_deg, order=1)
    volume_filtre = np.zeros((N_inv, A, B), dtype=np.float32)

    for n in range(N_inv):
        actifs = mask_2d[n] > 0.5  # [B]
        if not actifs.any():
            continue
        colonne_slice = rotated_img[n]  # [A, B]
        masque_hu = colonne_slice >= seuil_hu_metal
        masque_hu = masque_hu & actifs[np.newaxis, :]  # restreindre aux b actifs
        volume_filtre[n] = masque_hu.astype(np.float32)

    rotated_back = _rotate_plane_stack(volume_filtre, -theta_deg, order=0)
    rotated_back = np.nan_to_num(rotated_back, nan=0.0)
    volume_moved = (rotated_back > 0.5).astype(np.uint8)
    return np.moveaxis(volume_moved, 0, inv_idx)


# --------------------------------------------------------------------------- #
# Dispatcher : une seule entrée pour les 3 backprojections "par angle"
# (space_carving a une granularité différente — tous les angles à la
# fois — gérée séparément dans reconstruct_3d, pas ici)
# --------------------------------------------------------------------------- #

def backproject(
    method: Literal["tube_plein", "argmax_filtre", "seuil_hu"],
    mask_2d: np.ndarray,
    theta_deg: float,
    volume_shape_zyx: tuple[int, int, int],
    axe_invariant: str = "Z",
    image_3d: sitk.Image | None = None,
    alpha: float = 100.0,
    seuil_hu_metal: float = 1500.0,
) -> np.ndarray:
    """
    image_3d n'est utilisée que par argmax_filtre/seuil_hu ; tube_plein
    l'ignore — sa signature réelle (retroprojeter_masque) n'en a jamais eu
    besoin, ce n'est pas un oubli (voir docstring de module : tube_plein
    est purement géométrique).
    """
    if method == "tube_plein":
        return retroprojeter_masque(mask_2d, theta_deg, volume_shape_zyx, axe_invariant)
    if method == "argmax_filtre":
        if image_3d is None:
            raise ValueError("method='argmax_filtre' nécessite image_3d.")
        return backproject_argmax_filtre(mask_2d, image_3d, theta_deg, axe_invariant, alpha=alpha)
    if method == "seuil_hu":
        if image_3d is None:
            raise ValueError("method='seuil_hu' nécessite image_3d.")
        return backproject_seuil_hu(
            mask_2d, image_3d, theta_deg, axe_invariant, seuil_hu_metal=seuil_hu_metal
        )
    raise ValueError(f"méthode inconnue pour backproject : '{method}'.")


# --------------------------------------------------------------------------- #
# Méthode 4 — Space Carving (photo-consistency multi-vues, itératif)
# --------------------------------------------------------------------------- #

def space_carving(
    image_3d: sitk.Image,
    masks_2d: dict[float, np.ndarray],
    axe_invariant: str = "Z",
    seuil_hu_metal: float = 1500.0,
    vote_ratio: float = 1.0,
) -> np.ndarray:
    """
    Sculpture voxel par voxel : un voxel 3D est conservé seulement s'il
    est à la fois (a) dans la silhouette 2D à CHAQUE angle (comme le
    Visual Hull), ET (b) son intensité HU réelle dépasse le seuil métal.

    Contrairement à backproject_seuil_hu (qui filtre indépendamment par
    angle avant d'accumuler), ici on combine géométrie (silhouette
    multi-angles) et photo-consistency (HU réel du voxel) en un seul
    critère conjoint, évalué une fois pour chaque voxel du volume — c'est
    la définition propre du space carving.

    vote_ratio=1.0 -> voxel gardé seulement s'il est cohérent avec TOUS
                      les angles (strict, comme l'ancien mode "intersection").
    vote_ratio<1.0 -> tolère quelques angles incohérents.
    """
    array_hu = sitk.GetArrayFromImage(image_3d).astype(np.float32)  # (Z,Y,X)
    volume_shape_zyx = array_hu.shape
    n_angles = len(masks_2d)

    # Critère photométrique : le voxel doit être un métal plausible
    masque_hu_global = (array_hu >= seuil_hu_metal).astype(np.float32)

    # Critère géométrique : accumulation de silhouettes (Visual Hull
    # classique — méthode "tube_plein")
    accumulator = np.zeros(volume_shape_zyx, dtype=np.float32)
    for theta, mask_2d in masks_2d.items():
        accumulator += backproject("tube_plein", mask_2d, theta, volume_shape_zyx, axe_invariant)

    geometrie_ok = (accumulator >= vote_ratio * n_angles).astype(np.float32)

    # Space carving = intersection des deux critères
    volume_sculpte = (geometrie_ok * masque_hu_global > 0.5).astype(np.uint8)
    return volume_sculpte


# --------------------------------------------------------------------------- #
# Reconstruction 3D "de production" — une seule méthode, tous les angles
# --------------------------------------------------------------------------- #

def reconstruct_3d(
    masks_2d: dict[float, np.ndarray],
    reference_image: sitk.Image,
    axe_invariant: str = "Z",
    method: Literal["tube_plein", "argmax_filtre", "seuil_hu", "space_carving"] = "tube_plein",
    vote_ratio: float = 1.0,
    alpha_argmax: float = 100.0,
    seuil_hu_metal: float = 1500.0,
) -> sitk.Image:
    """
    Reconstruit le volume 3D à partir des masques 2D obtenus à plusieurs
    angles, dans le repère spatial de `reference_image` (spacing/origine/
    direction copiés sur le résultat via CopyInformation).

    `reference_image` sert AUSSI de source d'intensités HU pour les
    méthodes qui en ont besoin (argmax_filtre / seuil_hu / space_carving)
    — passer la vraie image CT dès que method != "tube_plein", pas un
    masque ou un volume vide (voir evaluate_reconstruction_on_dataset
    pour l'erreur que ça peut provoquer si on l'oublie).

    method : technique de backprojection à utiliser (les 4 comparées par
             comparer_methodes — brancher ici celle retenue une fois la
             comparaison faite sur un ou plusieurs cas de validation).
    vote_ratio : proportion minimale d'angles où le voxel doit être actif
                 pour être conservé.
                 1.0 = accord de TOUS les angles requis (ancien mode
                 "intersection" de ce module — pas un mode à part :
                 l'accumulateur ne dépasse jamais n_angles, donc
                 `accumulator == n_angles` équivaut exactement à
                 `accumulator >= 1.0 * n_angles`).
                 <1.0 = tolère des angles incohérents ; plus robuste à une
                 segmentation 2D isolée ratée, mais tend à sur-segmenter
                 si trop bas — vérifier empiriquement le Dice.
    """
    if method == "space_carving":
        recon = space_carving(
            reference_image, masks_2d, axe_invariant,
            seuil_hu_metal=seuil_hu_metal, vote_ratio=vote_ratio,
        )
    else:
        volume_shape_zyx = sitk.GetArrayFromImage(reference_image).shape
        n_angles = len(masks_2d)
        accumulator = np.zeros(volume_shape_zyx, dtype=np.float32)
        for theta, mask_2d in masks_2d.items():
            accumulator += backproject(
                method, mask_2d, theta, volume_shape_zyx, axe_invariant,
                image_3d=reference_image, alpha=alpha_argmax, seuil_hu_metal=seuil_hu_metal,
            )
        recon = (accumulator >= vote_ratio * n_angles).astype(np.uint8)

    recon_image = sitk.GetImageFromArray(recon)
    recon_image.CopyInformation(reference_image)
    return recon_image


# --------------------------------------------------------------------------- #
# Harnais de comparaison sur le dataset TotalSegmentator
# --------------------------------------------------------------------------- #

def comparer_methodes(
    image_3d: sitk.Image,
    gt_mask_3d: sitk.Image,
    masks_2d_par_angle: dict[float, np.ndarray],
    axe_invariant: str = "Z",
    alpha_argmax: float = 100.0,
    seuil_hu_metal: float = 1500.0,
    vote_ratio: float = 0.66,
) -> pandas.DataFrame:
    """
    Compare les 4 méthodes sur UN cas (un patient TotalSegmentator), en
    utilisant les mêmes masks_2d_par_angle (supposés être la vérité
    terrain projetée, ou les prédictions nnU-Net selon ce que vous testez).

    Délègue à reconstruct_3d pour chacune des 4 méthodes — ne duplique
    plus la logique de combinaison par angle, qui vit désormais à un seul
    endroit (backproject + reconstruct_3d).

    Métriques (dice_score/iou_score/hausdorff_distance) importées depuis
    src.evaluation.metrics — hausdorff_distance y est restreinte au
    contour du masque, ne pas la recalculer localement (voir docstring
    de module : une version calculée sur tous les voxels positifs
    désavantage artificiellement les méthodes qui sur-segmentent, comme
    tube_plein, et fausserait la comparaison).
    """
    import pandas as pd

    # spacing en ordre SimpleITK natif (X, Y, Z) — hausdorff_distance
    # l'inverse en interne pour l'aligner sur l'ordre du tableau numpy
    # (Z, Y, X). Ne pas inverser ici.
    spacing_xyz = image_3d.GetSpacing()
    gt_array = sitk.GetArrayFromImage(gt_mask_3d).astype(np.uint8)

    resultats = []
    for methode in ("tube_plein", "argmax_filtre", "seuil_hu", "space_carving"):
        recon_image = reconstruct_3d(
            masks_2d_par_angle, image_3d, axe_invariant=axe_invariant,
            method=methode, vote_ratio=vote_ratio,
            alpha_argmax=alpha_argmax, seuil_hu_metal=seuil_hu_metal,
        )
        pred = sitk.GetArrayFromImage(recon_image)
        resultats.append({
            "methode": methode,
            "dice": dice_score(pred, gt_array),
            "iou": iou_score(pred, gt_array),
            "hausdorff95_mm": hausdorff_distance(pred, gt_array, spacing=spacing_xyz, percentile=95.0),
        })

    return pd.DataFrame(resultats)


# --------------------------------------------------------------------------- #
# Évaluation de reconstruct_3d (une seule méthode, dataset complet)
#
# À distinguer de comparer_methodes : celle-ci évalue les 4 méthodes sur
# UN cas ; evaluate_reconstruction_on_dataset évalue LA méthode retenue
# sur un dataset entier.
# --------------------------------------------------------------------------- #

def evaluate_reconstruction_on_dataset(
    cases: list[dict],
    axe_invariant: str = "Z",
    method: Literal["tube_plein", "argmax_filtre", "seuil_hu", "space_carving"] = "tube_plein",
    vote_ratio: float = 1.0,
) -> pandas.DataFrame:
    """
    Évalue le pipeline complet (projection -> segmentation 2D -> reconstruction)
    sur un ensemble de cas disposant d'une vérité terrain 3D.

    cases : liste de dicts {
        "patient_id": str,
        "gt_mask_image": sitk.Image,          # vérité terrain 3D (référence spatiale)
        "predicted_masks_2d": dict[float, np.ndarray],  # sorties nnU-Net 2D par angle
        "ct_image": sitk.Image,               # image CT réelle — REQUISE dès que
                                               # method != "tube_plein" (seule méthode
                                               # purement géométrique qui n'a besoin
                                               # d'aucune intensité HU réelle).
    }
    L'inférence nnU-Net elle-même (génération de predicted_masks_2d) dépend
    de ton environnement et n'est pas incluse ici.

    Retourne un DataFrame avec Dice / IoU / Hausdorff95 par patient + moyennes.
    """
    import pandas as pd

    rows = []
    for case in cases:
        gt_image = case["gt_mask_image"]
        gt_array = sitk.GetArrayFromImage(gt_image).astype(np.uint8)
        # hausdorff_distance (src.evaluation.metrics) attend le spacing en
        # ordre SimpleITK natif (X, Y, Z) — elle l'inverse en interne pour
        # l'aligner sur l'ordre du tableau numpy (Z, Y, X). Ne PAS inverser
        # ici, sinon double-inversion et distances à nouveau fausses.
        spacing_xyz = gt_image.GetSpacing()

        if method == "tube_plein":
            reference_image = gt_image  # purement géométrique, pas besoin des HU
        else:
            if "ct_image" not in case:
                raise ValueError(
                    f"method='{method}' nécessite case['ct_image'] (image CT réelle) "
                    f"pour le patient {case.get('patient_id')} — gt_mask_image seul "
                    f"ne contient pas d'intensités HU."
                )
            reference_image = case["ct_image"]

        recon_image = reconstruct_3d(
            case["predicted_masks_2d"], reference_image,
            axe_invariant=axe_invariant, method=method, vote_ratio=vote_ratio,
        )
        recon_array = sitk.GetArrayFromImage(recon_image)

        hd95 = hausdorff_distance(recon_array, gt_array, spacing=spacing_xyz, percentile=95.0)

        rows.append({
            "patient_id": case["patient_id"],
            "n_angles": len(case["predicted_masks_2d"]),
            "dice": dice_score(recon_array, gt_array),
            "iou": iou_score(recon_array, gt_array),
            # inf (masque vide) -> None : pandas .mean() ignore les NaN/None
            # mais PAS inf, qui contaminerait la moyenne du dataset entier.
            "hausdorff95_mm": hd95 if not np.isinf(hd95) else None,
        })

    df = pd.DataFrame(rows)
    summary = df[["dice", "iou", "hausdorff95_mm"]].mean().to_dict()
    print("--- Moyennes sur le dataset ---")
    for k, v in summary.items():
        print(f"{k:>15} : {v:.4f}")
    return df


# --------------------------------------------------------------------------- #
# Exemple d'utilisation sur un cas TotalSegmentator
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from roi_projection import projeter

    image_3d = sitk.ReadImage("roi_ct.nii.gz")
    gt_mask_3d = sitk.ReadImage("roi_mask_totalsegmentator.nii.gz")

    angles = [0, 30, 60, 90, 120, 150]

    # Ici, remplacer par les VRAIES sorties nnU-Net 2D pour chaque angle.
    # Pour un premier test de la reconstruction seule (sans erreur de
    # segmentation 2D), on peut utiliser la projection du GT lui-même :
    masks_2d_par_angle = {}
    for theta in angles:
        proj_gt = projeter(gt_mask_3d, theta, axe_invariant="Z", is_mask=True)
        masks_2d_par_angle[theta] = sitk.GetArrayFromImage(proj_gt)

    df = comparer_methodes(image_3d, gt_mask_3d, masks_2d_par_angle,
                            axe_invariant="Z", alpha_argmax=100.0,
                            seuil_hu_metal=1500.0, vote_ratio=0.66)
    print(df)

    # Une fois la meilleure méthode identifiée dans df (ex. "space_carving") :
    # recon_finale = reconstruct_3d(masks_2d_par_angle, image_3d,
    #                                method="space_carving", vote_ratio=0.66,
    #                                seuil_hu_metal=1500.0)
    # sitk.WriteImage(recon_finale, "reconstruction_3d.nii.gz")