"""
roi_projection.py

Projection multi-angles d'un volume 3D vers des vues 2D synthétiques
(Stage 4 — segmentation de prothèse par vues synthétiques + Visual Hull).

Ce module ne fait QUE de la projection (3D -> 2D) et sa préparation. La
reconstruction (2D -> 3D, l'opération inverse) vit dans
reconstruction_methods.py — voir ARCHITECTURE.md §2 pour la séparation
entre les deux étapes du pipeline.

Brique centrale : `projeter`, qui projette un volume 3D sur un plan
(axe_invariant, D) où D est une droite du plan formé par les deux autres
axes, faisant un angle theta_deg avec le premier d'entre eux. Basée sur un
échantillonnage inverse (map_coordinates) plutôt qu'une rotation 3D +
collapse séparés :
  - pas de croissance de canvas (taille de sortie fixe, contrairement à
    scipy.ndimage.rotate(..., reshape=True)) ;
  - hors-volume marqué NaN (pas 0) pour ne pas fausser mean/min aux bords ;
  - pas de rééchantillonnage isotrope inutile : seul le plan tourné doit
    être isotrope, ce qui est déjà presque toujours le cas du plan (X,Y)
    en CT clinique quand axe_invariant="Z".

`_rotate_plane_stack` (noyau géométrique de rotation) reste ici bien
qu'il soit aussi utilisé par reconstruction_methods.py (qui l'importe) :
c'est la brique dont `projeter` lui-même dépend directement, donc la
séparation projection/reconstruction porte sur les fonctions de haut
niveau (retroprojeter_masque, reconstruct_3d, ...), pas sur ce noyau
partagé — à ne pas dupliquer côté reconstruction.

Convention DICOM/NIfTI usuelle : X=Gauche-Droite, Y=Antéro-Postérieur,
Z=Supéro-Inférieur (axe long du corps). axe_invariant="Z" est le cas
d'usage principal (vues face/profil autour de l'axe long).
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import map_coordinates


AXIS_NAMES = ["Z", "Y", "X"]
AggMethod = Union[str, Callable]


# --------------------------------------------------------------------------- #
# Vérification d'isotropie du plan de rotation
# --------------------------------------------------------------------------- #

def _plane_spacing(spacing_xyz: tuple[float, float, float], axe_invariant: str) -> tuple[float, float]:
    sx, sy, sz = spacing_xyz
    return {"Z": (sx, sy), "Y": (sx, sz), "X": (sy, sz)}[axe_invariant]


def _check_plane_isotropy(spacing_xyz: tuple[float, float, float], axe_invariant: str,
                           rtol: float = 1e-2, strict: bool = True) -> float:
    """
    Vérifie que le plan qui va être tourné (les deux axes autres que
    axe_invariant) a un spacing isotrope. Si ce n'est pas le cas, la
    rotation en indices de voxels (sans passer par des mm) est
    géométriquement fausse : un cercle deviendrait une ellipse selon theta.

    Retourne le spacing commun du plan (réutilisé comme spacing physique
    de l'axe D en sortie de projeter_volume).
    """
    s1, s2 = _plane_spacing(spacing_xyz, axe_invariant)
    if not np.isclose(s1, s2, rtol=rtol):
        msg = (f"Plan de rotation non isotrope pour axe_invariant='{axe_invariant}' : "
               f"spacing=({s1:.3f}, {s2:.3f}) mm. La rotation en indices de voxels sera "
               f"géométriquement faussée (un cercle deviendrait une ellipse).")
        if strict:
            raise ValueError(msg + " Rééchantillonne ce plan en isotrope avant d'appeler "
                                    "projeter, ou passe check_isotropy=False pour accepter "
                                    "ce compromis (déconseillé).")
        else:
            import warnings
            warnings.warn(msg)
    return float((s1 + s2) / 2.0)


# --------------------------------------------------------------------------- #
# Cœur géométrique partagé : rotation par échantillonnage inverse
# --------------------------------------------------------------------------- #

def _rotate_plane_stack(array_moved: np.ndarray, theta_deg: float, order: int) -> np.ndarray:
    """
    Fait tourner chaque tranche du stack (N_inv, A, B) de theta_deg dans le
    plan (A, B), par échantillonnage inverse (map_coordinates), en gardant
    la taille (A, B) fixe (pas de croissance de canvas contrairement à
    scipy.ndimage.rotate(reshape=True)).

    Hors-volume marqué NaN, pas 0, pour ne pas fausser une agrégation
    mean/min ultérieure avec un padding artificiel.

    Utilisée à la fois par projeter (theta_deg > 0, projection) et par
    retroprojeter_masque (theta_deg < 0, rétroprojection) — même
    transformation géométrique, sens opposé.

    NOTE : bien que préfixée `_` (usage interne à ce module), cette
    fonction est aussi importée par reconstruction_methods.py — c'est le
    seul noyau géométrique de rotation du projet, à ne pas dupliquer.
    """
    N_inv, A, B = array_moved.shape
    theta = np.deg2rad(theta_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    ac, bc = (A - 1) / 2.0, (B - 1) / 2.0

    a_out, b_out = np.meshgrid(np.arange(A), np.arange(B), indexing="ij")
    a_rel, b_rel = a_out - ac, b_out - bc
    a_src = ac + a_rel * cos_t - b_rel * sin_t
    b_src = bc + a_rel * sin_t + b_rel * cos_t
    coords = np.stack([a_src, b_src], axis=0)

    rotated = np.empty_like(array_moved, dtype=np.float32)
    for n in range(N_inv):
        rotated[n] = map_coordinates(
            array_moved[n].astype(np.float32), coords, order=order,
            mode="constant", cval=np.nan,
        )
    return rotated


# --------------------------------------------------------------------------- #
# Projection (projeter, intégrée telle quelle + garde-fous)
# --------------------------------------------------------------------------- #

def projeter(
    image_3d: sitk.Image,
    theta_deg: float,
    axe_invariant: str = "Z",
    slab_min: Optional[int] = None,
    slab_max: Optional[int] = None,
    agregation: AggMethod = "max",
    order: int = 1,
    is_mask: bool = False,
    check_isotropy: bool = True,
) -> sitk.Image:
    """
    Projette un volume 3D (Z, Y, X) sur un plan (axe_invariant, D), D étant
    une droite du plan formé par les deux autres axes, faisant un angle
    `theta_deg` avec le premier d'entre eux.

    axe_invariant="Z" (par défaut) : rotation dans le plan (Y, X).
        theta=0°  -> agrégation sur Y -> vue de FACE (coronale)
        theta=90° -> agrégation sur X -> vue de PROFIL (sagittale)
    axe_invariant="Y" : rotation dans le plan (Z, X) (vue axiale à theta=0°).
    axe_invariant="X" : rotation dans le plan (Z, Y).

    is_mask=True force order=0 et agregation="max" : indispensable pour
    une vérité terrain, afin qu'elle reste binaire quel que soit ce qui
    est demandé pour l'image associée.

    check_isotropy=True (par défaut) lève une erreur si le plan tourné
    n'est pas isotrope — la rotation en indices de voxels serait sinon
    silencieusement fausse. Ne désactive ce garde-fou que si tu as déjà
    vérifié/rééchantillonné ce plan par ailleurs.

    slab_min/slab_max restreignent l'axe de profondeur (le premier des
    deux axes tournés) avant agrégation, en indices de voxel : plage
    complète = projection (MIP/mean/min plein volume), plage étroite =
    section fine (reproduit une coupe CT réelle à cet angle).
    """
    if axe_invariant not in AXIS_NAMES:
        raise ValueError(f"axe_invariant doit être 'Z', 'Y' ou 'X', reçu '{axe_invariant}'.")
    inv_idx = AXIS_NAMES.index(axe_invariant)

    if is_mask:
        order = 0
        agregation = "max"

    spacing_xyz = image_3d.GetSpacing()
    d_axis_spacing = _check_plane_isotropy(spacing_xyz, axe_invariant, strict=check_isotropy)

    array = sitk.GetArrayFromImage(image_3d).astype(np.float32)  # (Z, Y, X)
    array_moved = np.moveaxis(array, inv_idx, 0)                  # (N_inv, A, B)

    rotated = _rotate_plane_stack(array_moved, theta_deg, order=order)
    rotated_slab = rotated[:, slab_min:slab_max, :]

    agg_map = {"max": np.nanmax, "mean": np.nanmean, "min": np.nanmin}
    if agregation in agg_map:
        agg_func = agg_map[str(agregation)]
    elif callable(agregation):
        agg_func = agregation
    else:
        raise ValueError(f"Agrégation '{agregation}' non gérée.")

    projection = agg_func(rotated_slab, axis=1)
    projection = np.nan_to_num(projection, nan=0.0, posinf=0.0, neginf=0.0)

    image_2d = sitk.GetImageFromArray(projection.astype(np.float32))
    # Axe D (colonnes) : spacing physique réel du plan isotrope tourné
    # (une rotation euclidienne préserve les longueurs dans un plan isotrope).
    # Axe invariant (lignes) : son propre spacing d'origine, inchangé.
    spacing_orig = image_3d.GetSpacing()
    spacing_par_axe = {"X": spacing_orig[0], "Y": spacing_orig[1], "Z": spacing_orig[2]}
    image_2d.SetSpacing((d_axis_spacing, spacing_par_axe[axe_invariant]))
    return image_2d


# --------------------------------------------------------------------------- #
# Marge de sécurité avant projection (évite le clipping aux coins)
# --------------------------------------------------------------------------- #

def pad_roi_margin(image_3d: sitk.Image, margin_mm: float) -> sitk.Image:
    """
    Ajoute une marge de fond (zéro) autour de la ROI avant projection.

    projeter ne fait pas grossir le canvas (contrairement à
    scipy.ndimage.rotate(reshape=True)) : un objet proche du bord de la
    ROI peut donc être tronqué aux coins lors d'une rotation à 45°/135°
    (la diagonale atteint jusqu'à √2 fois la demi-largeur). Appelle cette
    fonction sur la ROI avant projeter si la prothèse n'est pas bien
    centrée avec une marge suffisante.
    """
    spacing = image_3d.GetSpacing()  # (sx, sy, sz)
    margin_voxels = [int(np.ceil(margin_mm / s)) for s in spacing]
    pad_filter = sitk.ConstantPadImageFilter()
    pad_filter.SetPadLowerBound(margin_voxels)
    pad_filter.SetPadUpperBound(margin_voxels)
    pad_filter.SetConstant(0)
    return pad_filter.Execute(image_3d)


# --------------------------------------------------------------------------- #
# Génération du dataset multi-angles (image + masque), pour l'entraînement
# --------------------------------------------------------------------------- #

def generate_multiangle_pairs(
    image_3d: sitk.Image,
    mask_3d: sitk.Image,
    angles_deg: list[float] | list[int],
    axe_invariant: str = "Z",
    slab_min: Optional[int] = None,
    slab_max: Optional[int] = None,
    agregation: AggMethod = "max",
    order: int = 1,
) -> list[dict]:
    """
    Génère les paires (image_2D, masque_2D) pour chaque angle, prêtes à
    être sauvées comme dataset d'entraînement nnU-Net 2D (Stage 4).

    Le masque est toujours projeté avec is_mask=True (order=0, max forcé),
    indépendamment de agregation/order demandés pour l'image.
    """
    pairs = []
    for theta in angles_deg:
        img_2d = projeter(
            image_3d, theta, axe_invariant, slab_min, slab_max,
            agregation=agregation, order=order, is_mask=False,
        )
        mask_2d = projeter(
            mask_3d, theta, axe_invariant, slab_min, slab_max,
            is_mask=True,
        )
        pairs.append({
            "theta": theta,
            "image": sitk.GetArrayFromImage(img_2d),
            "mask": sitk.GetArrayFromImage(mask_2d).astype(np.uint8),
        })
    return pairs


# --------------------------------------------------------------------------- #
# Exemple d'utilisation
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    image_3d = pad_roi_margin(sitk.ReadImage("roi_ct.nii.gz"), margin_mm=10.0)
    mask_3d = pad_roi_margin(sitk.ReadImage("roi_mask_totalsegmentator.nii.gz"), margin_mm=10.0)

    angles = [0, 30, 60, 90, 120, 150]

    # Génération du dataset d'entraînement 2D (projections MIP)
    pairs = generate_multiangle_pairs(image_3d, mask_3d, angles, agregation="max")
    for p in pairs:
        print(f"theta={p['theta']:>4}°  image {p['image'].shape}  "
              f"masque {p['mask'].shape}  pixels positifs={p['mask'].sum()}")

    # --- Après inférence nnU-Net (hors de ce script) sur chaque projection,
    #     la reconstruction 3D (retroprojeter_masque / reconstruct_3d /
    #     evaluate_reconstruction_on_dataset) vit dans
    #     reconstruction_methods.py — pas dans ce module, dédié à la
    #     projection uniquement. Voir ce fichier pour l'exemple d'usage.