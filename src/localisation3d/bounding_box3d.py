# src/localisation3d/bounding_box3d.py

from __future__ import annotations
import logging
import SimpleITK as sitk

from src.utils.io import extraire_bboxes, calculer_lateralite

logger = logging.getLogger(__name__)





def extraire_bboxes_coronale(mask_2d: sitk.Image, longueur_x: int) -> dict[str, list]:
    """
    Extrait les BBox 2D depuis le masque coronal (face), en séparant
    gauche/droite par la position du centroïde de chaque composante
    connexe. Gère nativement le cas de deux prothèses distinctes.

    Retourne : {"gauche": [xmin, zmin, x_len, z_len] ou [],
                "droite": [xmin, zmin, x_len, z_len] ou []}
    """
    MAP_LATERALITE_COTE = {1: "gauche", 2: "droite"}
    resultat = {"gauche": [], "droite": []}

    bboxes_list = extraire_bboxes(mask_2d)

    for bbox in bboxes_list:
        lateralite = calculer_lateralite([bbox], longueur_x=longueur_x, est_profil=False)
        cote = MAP_LATERALITE_COTE.get(lateralite)
        if cote is None:
            # lateralite == 0 : la composante chevauche la ligne médiane,
            # ni entièrement à gauche ni entièrement à droite -> écartée.
            logger.warning(
                f"Composante coronale écartée (chevauche la ligne médiane) : bbox={bbox}"
            )
            continue

        if resultat[cote]:
            # Si déjà occupé (cas rare de 2 composantes du même côté,
            # ex: artefact) -> fusionner par union
            ancien = resultat[cote]
            nouveau_xmin = min(ancien[0], bbox[0])
            nouveau_zmin = min(ancien[1], bbox[1])
            nouveau_xmax = max(ancien[0] + ancien[2], bbox[0] + bbox[2])
            nouveau_zmax = max(ancien[1] + ancien[3], bbox[1] + bbox[3])
            resultat[cote] = [
                nouveau_xmin, nouveau_zmin,
                nouveau_xmax - nouveau_xmin, nouveau_zmax - nouveau_zmin
            ]
        else:
            resultat[cote] = list(bbox)

    return resultat


def extraire_bbox_sagittale(mask_2d: sitk.Image) -> list:
    """
    Extrait la BBox 2D depuis un masque sagittal (profil, déjà restreint
    à une moitié du corps) — on attend au plus une seule composante
    pertinente ; si plusieurs, on prend l'union (cas prothèse + artefact
    proche, rare mais possible).
    """
    bboxes_list = extraire_bboxes(mask_2d)
    resultat = []

    for bbox in bboxes_list:
        if resultat != []:
                ancien = resultat
                nouveau_ymin = min(ancien[0], bbox[0])
                nouveau_zmin = min(ancien[1], bbox[1])
                nouveau_ymax = max(ancien[0] + ancien[2], bbox[0] + bbox[2])
                nouveau_zmax = max(ancien[1] + ancien[3], bbox[1] + bbox[3])
                resultat = [
                    nouveau_ymin, nouveau_zmin,
                    nouveau_ymax - nouveau_ymin, nouveau_zmax - nouveau_zmin
                ]
        else:
            resultat = list(bbox)

    return resultat

def get_3d_bboxes(dict_2d_bboxes: dict[str, tuple[list, list]],
                   mode: str = "union") -> dict[str, list]:
    """
    Construit une BBox 3D par côté (gauche/droite) à partir des paires
    (bbox coronale, bbox sagittale) déjà assemblées par l'appelant.

    Un côté sans prothèse (bbox coronale et/ou sagittale absente, ex:
    prothèse unilatérale — le cas le plus fréquent) est ignoré proprement
    plutôt que de faire planter l'unpacking dans _2d_bboxes_to_3d.

    Args:
        dict_2d_bboxes: {"gauche": (bbox_coronale, bbox_sagittale),
                         "droite": (bbox_coronale, bbox_sagittale)},
                        chaque bbox pouvant être [] si absente pour ce côté.
        mode:           "union" ou "intersection", voir _2d_bboxes_to_3d.

    Returns:
        {"gauche": bbox_3d | [], "droite": bbox_3d | []}.
    """
    dict_3d_bboxes = {}
    for side, bboxes in dict_2d_bboxes.items():
        bbox_coronale, bbox_sagittale = bboxes
        if not bbox_coronale or not bbox_sagittale:
            logger.info(f"Côté '{side}' sans BBox 3D (coronale et/ou sagittale absente)")
            dict_3d_bboxes[side] = []
            continue
        dict_3d_bboxes[side] = _2d_bboxes_to_3d(bboxes, mode=mode)
    return dict_3d_bboxes



def _2d_bboxes_to_3d(tuple_of_2d_bboxes: tuple[list, list], mode: str = "union") -> list:
    """
    Fusionne une BBox 2D coronale (x, z) et une BBox 2D sagittale (y, z)
    en une BBox 3D (x, y, z), en réconciliant les deux estimations de z
    de façon géométriquement cohérente.

    mode="union"        : zmin = min des deux, zmax = max des deux
                          (plus permissif, ne rate jamais la zone)
    mode="intersection" : zmin = max des deux, zmax = min des deux
                          (plus précis, mais peut être vide si désaccord fort)
    Args:
        tuple_of_2d_bboxes: un tuple contenant une BBox 2D coronale (x, z) et une BBox 2D sagittale (y, z) dans l'ordre
    Return:
        list avec les caractéristiques de la bounding box 3D [xmin, ymin, zmin, x_len, y_len, z_len_final]
    """
    xmin, zmin_coronal, x_len, z_len_coronal = tuple_of_2d_bboxes[0]
    ymin, zmin_sagital, y_len, z_len_sagital = tuple_of_2d_bboxes[1]

    zmax_coronal = zmin_coronal + z_len_coronal
    zmax_sagital = zmin_sagital + z_len_sagital

    if mode == "union":
        zmin_final = min(zmin_coronal, zmin_sagital)
        zmax_final = max(zmax_coronal, zmax_sagital)
    elif mode == "intersection":
        zmin_final = max(zmin_coronal, zmin_sagital)
        zmax_final = min(zmax_coronal, zmax_sagital)
        if zmax_final <= zmin_final:
            raise ValueError(
                f"Intersection Z vide : coronale z∈[{zmin_coronal},{zmax_coronal}], "
                f"sagittale z∈[{zmin_sagital},{zmax_sagital}] ne se recouvrent pas."
            )
    else:
        raise ValueError(f"mode '{mode}' inconnu, choisir 'union' ou 'intersection'.")

    z_len_final = zmax_final - zmin_final

    return [xmin, ymin, zmin_final, x_len, y_len, z_len_final]