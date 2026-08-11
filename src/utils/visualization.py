from ..utils.paths import FIGURES
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math


# --------------------------------------------------------------------------- #
# Conversion image -> array 2D (SimpleITK ou numpy, coupe 3D si besoin)
# --------------------------------------------------------------------------- #

def _to_array(img, slice_idx=None, axe=0):
    if isinstance(img, sitk.Image):
        is_rgb = img.GetNumberOfComponentsPerPixel() > 1
        arr = sitk.GetArrayFromImage(img)
    else:
        arr = np.asarray(img)
        is_rgb = arr.ndim >= 3 and arr.shape[-1] in [3, 4]

    if arr.ndim >= 3 and not is_rgb:
        idx = slice_idx if slice_idx is not None else arr.shape[axe] // 2
        if axe == 0:
            return arr[idx]
        elif axe == 1:
            return arr[:, idx, :]
        else:
            return arr[:, :, idx]
    elif arr.ndim == 4 and is_rgb:
        idx = slice_idx if slice_idx is not None else arr.shape[axe] // 2
        if axe == 0:
            return arr[idx, ..., :]
        elif axe == 1:
            return arr[:, idx, ..., :]
        else:
            return arr[:, :, idx, :]
    return arr


def _to_overlay(img, masque, vmin, vmax, slice_idx=None, axe=0, opacity=0.4):
    """
    Overlay masque/image avec fenêtrage explicite [vmin, vmax].

    Contrairement à sitk.RescaleIntensity(img) qui s'étire sur le
    min/max PROPRE de chaque image (contraste incohérent d'une image
    à l'autre), on clampe d'abord sur [vmin, vmax] — les mêmes valeurs
    que celles utilisées pour l'affichage des images brutes — pour que
    l'overlay ait un contraste cohérent avec le reste de la grille.
    """
    img_f       = sitk.Cast(img, sitk.sitkFloat32)
    img_clamped = sitk.Clamp(img_f, lowerBound=float(vmin), upperBound=float(vmax))
    img_u8      = sitk.Cast(sitk.RescaleIntensity(img_clamped, 0, 255), sitk.sitkUInt8)
    ov = sitk.LabelOverlay(
        img_u8,
        sitk.Cast(masque, sitk.sitkUInt8),
        opacity=opacity,
    )
    return _to_array(ov, slice_idx=slice_idx, axe=axe)


def _compute_vmin_vmax(images, slice_idx=None, axe=0):
    """
    Calcule un fenêtrage global (percentile 1-99) à partir des images
    brutes, avant tout découpage en pages — pour que le contraste soit
    cohérent sur toute la grille plutôt que recalculé page par page.
    """
    raw_arrays = [_to_array(img, slice_idx=slice_idx, axe=axe) for img in images]
    n = len(raw_arrays)
    sample = raw_arrays[::max(1, n // 20)] if n > 20 else raw_arrays
    flat = np.concatenate([
        s[..., 0].ravel() if s.ndim == 3 else s.ravel()
        for s in sample
    ])
    return float(np.percentile(flat, 1)), float(np.percentile(flat, 99))


def _bbox_coords(box, ndim):
    """
    Extrait (x, y, w, h) depuis GetBoundingBox selon la dimension.
        2D → (x, y, w, h)       indices (0, 1, 2, 3)
        3D → (x, y, z, w, h, d) indices (0, 1, 2, 3, 4, 5)
             on projette en XY  → (x, y, w, h) = (0, 1, 3, 4)
    """
    if ndim == 2 or len(box) == 4:
        return box[0], box[1], box[2], box[3]
    else:
        return box[0], box[1], box[3], box[4]


# --------------------------------------------------------------------------- #
# preparer_cases — construit une liste générique de "cases à afficher"
# --------------------------------------------------------------------------- #

def preparer_cases(
    images,
    titres=None,
    overlays=None,      # list[sitk.Image | list[sitk.Image] | None] par image
    boxes_list=None,     # list[tuple(box, ndim) | list[box] | None] par image
    slice_idx=None,
    axe=0,
    vmin=None,
    vmax=None,
    overlay_opacity=0.4,
    overlay_suffixe="seg",
):
    """
    Construit une liste générique de "cases à afficher", chaque case étant
    un dict {array, titre, box}.

    Remplace la gestion en dur de 1 ou 2 masques (masques/masques2 +
    comparaison) par un paramètre unique `overlays` : pour chaque image,
    on peut passer None (pas d'overlay), un seul sitk.Image (1 overlay),
    ou une liste de sitk.Image (N overlays, un par segmentation à
    comparer) — sans jamais avoir besoin d'un nouveau paramètre
    masques3/comparaison2 si un 3e cas apparaît un jour.

    Le fenêtrage (vmin/vmax) est calculé UNE SEULE FOIS sur l'ensemble
    des images brutes si non fourni, puis appliqué de façon cohérente
    aux images brutes ET aux overlays (voir _to_overlay) — contraste
    global identique sur toute la grille plutôt que recalculé par case
    ou par page.

    Args:
        images          : list[sitk.Image | np.ndarray]
        titres          : list[str] optionnel, un par image
        overlays        : list optionnel, un élément par image :
                          None, un sitk.Image, ou une list[sitk.Image]
        boxes_list      : list optionnel, une bbox (ou liste de bbox) par
                          image — appliquée à CHAQUE overlay produit pour
                          cette image (masque 1 et masque 2 en mode
                          comparaison affichent tous les deux la bbox)
        slice_idx, axe  : coupe pour volumes 3D
        vmin, vmax      : fenêtrage explicite ; si None, calculé une fois
                          sur toutes les images (percentile 1-99)
        overlay_opacity : opacité de l'overlay masque
        overlay_suffixe : préfixe de titre pour les cases overlay

    Returns:
        cases : list[dict] {"array": np.ndarray, "titre": str,
                             "box": (box, ndim) | None}
        vmin, vmax : fenêtrage effectivement utilisé (à repasser tel
                     quel à render_page pour chaque page, afin de garder
                     un contraste identique sur toutes les pages)
    """
    if vmin is None or vmax is None:
        auto_vmin, auto_vmax = _compute_vmin_vmax(images, slice_idx=slice_idx, axe=axe)
        vmin = auto_vmin if vmin is None else vmin
        vmax = auto_vmax if vmax is None else vmax

    cases = []

    for i, img in enumerate(images):
        titre_i = titres[i] if titres and i < len(titres) else f"{i}"
        box_i = boxes_list[i] if boxes_list and i < len(boxes_list) else None
        ndim = img.GetDimension() if isinstance(img, sitk.Image) else 2

        overlay_i = overlays[i] if overlays and i < len(overlays) else None

        if overlay_i is None:
            # Pas d'overlay : image brute seule, avec box éventuelle
            cases.append({
                "array": _to_array(img, slice_idx=slice_idx, axe=axe),
                "titre": titre_i,
                "box": (box_i, ndim) if box_i else None,
            })
            continue

        # Normaliser overlay_i en liste (1 ou N overlays)
        liste_overlays = overlay_i if isinstance(overlay_i, (list, tuple)) else [overlay_i]

        # Image brute d'abord, sans box (comportement historique conservé)
        cases.append({
            "array": _to_array(img, slice_idx=slice_idx, axe=axe),
            "titre": titre_i,
            "box": None,
        })

        # Une case par overlay (1 masque, ou N masques à comparer) —
        # la box est dessinée sur CHAQUE overlay, pas seulement le dernier,
        # pour pouvoir comparer visuellement masque 1 et masque 2.
        n_overlays = len(liste_overlays)
        for j, masque in enumerate(liste_overlays):
            if masque is None:
                continue
            suffixe = overlay_suffixe if n_overlays == 1 else f"{overlay_suffixe}{j+1}"
            box_pour_cette_case = (box_i, ndim) if box_i else None
            cases.append({
                "array": _to_overlay(img, masque, vmin, vmax, slice_idx=slice_idx, axe=axe,
                                      opacity=overlay_opacity),
                "titre": f"{suffixe}_{titre_i}",
                "box": box_pour_cette_case,
            })

    return cases, vmin, vmax


# --------------------------------------------------------------------------- #
# render_page — rendu matplotlib pur, aucune logique métier
# --------------------------------------------------------------------------- #

def render_page(
    cases,
    page_num=0,
    n_pages=1,
    n_cols=None,
    taille_case=2.5,
    cmap='gray',
    vmin=None,
    vmax=None,
    titre_global=None,
    no_window=False,
    save_fig=False,
    batch_size=100,
):
    """
    Rend une page de cases (dicts {array, titre, box}) en grille matplotlib.
    Ne contient aucune logique de préparation de données — reçoit des
    cases déjà prêtes à afficher (voir preparer_cases).

    Args:
        cases        : list[dict] — une page de cases {array, titre, box}
        page_num     : index de la page courante (pour titre + nom fichier)
        n_pages      : nombre total de pages (pour affichage titre)
        n_cols       : colonnes (auto si None)
        taille_case  : taille en pouces par case
        cmap         : colormap matplotlib
        vmin/vmax    : fenêtrage (None = percentile 1-99 auto sur ces cases)
        titre_global : titre de la figure
        no_window    : pas de fenêtrage d'intensité pour l'affichage
        save_fig     : option de sauvegarde du graphique
        batch_size   : utilisé uniquement pour le nom de fichier de sauvegarde
    """
    n = len(cases)
    if n == 0:
        print("Aucune case à afficher sur cette page.")
        return

    arrays = [c["array"] for c in cases]
    titres = [c["titre"] for c in cases]
    boxes = [c["box"] for c in cases]

    # Fenêtrage local à la page si non fourni
    if vmin is None and vmax is None:
        sample = arrays[::max(1, n // 20)] if n > 20 else arrays
        flat = np.concatenate([
            s[..., 0].ravel() if s.ndim == 3 else s.ravel()
            for s in sample
        ])
        page_vmin = float(np.percentile(flat, 1))
        page_vmax = float(np.percentile(flat, 99))
    else:
        page_vmin, page_vmax = vmin, vmax

    cols = n_cols if n_cols else 2 * (min(10, max(1, math.ceil(math.sqrt(n * 4 / 3)))) // 2)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * taille_case, rows * taille_case),
        facecolor='#111111'
    )
    axes = np.array(axes).flatten()

    page_str = f" — page {page_num+1}/{n_pages}" if n_pages > 1 else ""
    fig.suptitle(f"{titre_global or ''}{page_str}", color='white', fontsize=10)

    for i, arr in enumerate(arrays):
        ax = axes[i]

        if arr.ndim == 3:
            ax.imshow(arr)
        elif no_window:
            ax.imshow(arr, cmap=cmap, aspect='equal', interpolation='nearest')
        else:
            ax.imshow(arr, cmap=cmap, vmin=page_vmin, vmax=page_vmax,
                      aspect='equal', interpolation='nearest')

        box_entry = boxes[i]
        if box_entry is not None:
            box, ndim = box_entry
            if box:
                for b in (box if isinstance(box[0], (list, tuple)) else [box]):
                    x, y, w, h = _bbox_coords(b, ndim)
                    ax.add_patch(patches.Rectangle(
                        (x, y), w, h,
                        linewidth=1, edgecolor='yellow', facecolor='none'
                    ))

        ax.set_title(titres[i], fontsize=20, color='#cccccc', pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')
            spine.set_linewidth(0.5)

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout(pad=0.2)
    if save_fig:
        FIGURES.mkdir(parents=True, exist_ok=True)
        chemin = FIGURES / f"{titre_global or 'figure'}_page_{page_num+1}_{n_pages}.png"
        fig.savefig(chemin, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"  💾 Sauvegardé : {chemin}")
    plt.show()


# --------------------------------------------------------------------------- #
# afficher_grille — thin wrapper, garde la compatibilité des appels existants
# --------------------------------------------------------------------------- #

def afficher_grille(
    images,
    titres=None,
    n_cols=None,
    taille_case=2.5,
    cmap='gray',
    slice_idx=None,
    axe=0,
    vmin=None,
    vmax=None,
    titre_global=None,
    batch_size=100,
    masques=None,          # optionnel : liste de masques à overlayer (compat.)
    boxes_list=None,       # optionnel : liste de bboxes par image
    interactive=False,
    no_window=False,
    save_fig=False,
    comparaison=False,     # conservé pour compatibilité des appels existants
    masques2=None,         # conservé pour compatibilité des appels existants
):
    """
    Affiche une grille adaptive d'images SimpleITK ou numpy.

    Thin wrapper : construit les cases via preparer_cases() puis délègue
    le rendu à render_page() pour chaque page. Toute nouvelle logique de
    préparation de données va dans preparer_cases, tout nouveau réglage
    d'affichage va dans render_page — cette fonction ne fait qu'assembler
    les deux et gérer le découpage en pages + la pause interactive.

    Args:
        images      : list[sitk.Image | np.ndarray]
        titres      : list[str] optionnel
        n_cols      : colonnes (auto si None)
        taille_case : taille en pouces par case
        cmap        : colormap matplotlib
        slice_idx   : coupe pour volumes 3D (None = centrale)
        axe         : axe de coupe 3D (0=axial, 1=coronal, 2=sagittal)
        vmin/vmax   : fenêtrage (None = percentile 1-99 auto)
        titre_global: titre de la figure
        batch_size  : images par page
        masques     : list[sitk.Image] masques à superposer (même longueur qu'images)
        boxes_list  : list[list[tuple]] bboxes par image
        interactive : pause entre pages
        no_window   : pas de fenetrage d'intensité pour l'affichage
        save_fig    : option de sauvegarde des graphiques
        comparaison : si True et masques2 fourni, affiche masques ET
                      masques2 comme deux overlays distincts (équivalent
                      à passer overlays=[[m1,m2], ...] à preparer_cases)
        masques2    : liste de masques additionnels si comparaison=True
    """
    # --- Reconstruction du paramètre unifié `overlays` depuis l'API
    #     historique (masques / masques2 / comparaison), pour compatibilité.
    overlays = None
    if masques is not None:
        if comparaison and masques2 is not None:
            overlays = [
                [m1,  masques2[i]] if (m1 is not None and i < len(masques2) and masques2[i] is not None) else m1
                for i, m1 in enumerate(masques)
            ]
        else:
            overlays = masques

    cases, vmin, vmax = preparer_cases(
        images,
        titres=titres,
        overlays=overlays,
        boxes_list=boxes_list,
        slice_idx=slice_idx,
        axe=axe,
        vmin=vmin,
        vmax=vmax,
    )

    n_total = len(cases)
    if n_total == 0:
        print("Aucune image à afficher.")
        return

    def make_batches(lst, size):
        return [lst[i:i + size] for i in range(0, len(lst), size)]

    case_batches = make_batches(cases, batch_size)
    n_pages = len(case_batches)

    print(f"{'─'*60}")
    print(f"  {len(images)} images  •  {n_total} cases  •  {n_pages} page(s)  •  {batch_size}/page")
    print(f"{'─'*60}")

    for p, batch in enumerate(case_batches):
        render_page(
            batch,
            page_num=p,
            n_pages=n_pages,
            n_cols=n_cols,
            taille_case=taille_case,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            titre_global=titre_global,
            no_window=no_window,
            save_fig=save_fig,
            batch_size=batch_size,
        )
        if interactive and p < n_pages - 1:
            rep = input(f"\n[Entrée] suite  |  [q] quitter  ({p+2}/{n_pages}) : ")
            if rep.strip().lower() == 'q':
                break