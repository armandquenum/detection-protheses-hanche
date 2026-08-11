# ─────────────────────────────────────────────
# src/detection/segmentation.py
# ─────────────────────────────────────────────
import logging
import SimpleITK as sitk
import numpy as np

from .constants import (
    SURFACE_MIN_MM2, SURFACE_MAX_MM2,
    ELONG_MIN, ELONG_MAX, LARGEUR_MIN_PX
)
from ..preprocessing.filters import seuil_par_percentile, nettoyage_image_rapide
from src.utils.io import calculer_lateralite

logger = logging.getLogger(__name__)


def get_segmentation(
    img: sitk.Image,
    est_profil: bool | None,
    afficher_intermediaire: bool = False,
    silent: bool = True,
    seuil_percent: float = 98,
) -> tuple[bool, sitk.Image, list, int]:
    """
    Détecte et segmente les prothèses de hanche dans un topogramme.

    Pipeline :
        1. Seuillage au percentile 98 → image binaire
        2. Fermetures morphologiques + nettoyage des pixels isolés
        3. Extraction des composantes connexes
        4. Filtrage par surface et élongation
        5. Validation par ConfidenceConnected sur le gradient

    Args:
        img:                    Image préprocessée (sortie de preprocessing_image).
        est_profil:             Booléen indiquant si l'image est en vue de profil (None si indéterminé)
        afficher_intermediaire: Afficher les étapes intermédiaires (debug).
        silent:                 Ne pas faire d'affichages
        seuil_percent:          seuil pour appliquer le filtre.

    Returns:
        prothese:     True si au moins une prothèse est détectée.
        masque_final: Masque binaire de la/les prothèse(s) détectée(s).
        bboxes:       Liste des bounding boxes (format GetBoundingBox).
        lateralite:   Latéralité de la prothèse :
                          -1 → orientation indéterminée (est_profil inconnu)
                           0 → pas de prothèse
                           1 → prothèse à gauche
                           2 → prothèse à droite
                           3 → prothèses bilatérales
                           4 → vue de profil, latéralité non calculable
    """
    sx, sy, sz = img.GetSpacing()
    longueur_x = img.GetSize()[0]
    prothese = False
    bboxes   = []
    masque_final = sitk.Image(img.GetSize(), sitk.sitkUInt8)
    masque_final.CopyInformation(img)

    # ── Étape 1 : binarisation par seuil haut ────────────────────────────
    seuil     = seuil_par_percentile(img, seuil_percent)
    image_u8  = sitk.Cast(img > seuil, sitk.sitkUInt8)
    

    # ── Étape 2 : nettoyage morphologique ────────────────────────────────
    image_u8 = sitk.GrayscaleConnectedClosing(image_u8)
    image_u8 = sitk.BinaryClosingByReconstruction(image_u8)
    image_u8 = nettoyage_image_rapide(image_u8)
    
    # Fermeture verticale uniquement [0,1,0] : les prothèses sont allongées
    # en hauteur → on connecte les fragments selon l'axe Y seulement
    image_u8 = sitk.BinaryMorphologicalClosing(image_u8, [0, 1, 0])

    if afficher_intermediaire:
        from ..utils.visualization import afficher_grille
        afficher_grille(
            [img, image_u8],
            ['Image brute', 'Image filtrée'],
            no_window=True
        )

    # ── Étape 3 : composantes connexes + tri par taille ──────────────────
    composantes       = sitk.ConnectedComponent(image_u8)
    composantes_tries = sitk.RelabelComponent(
        composantes,
        minimumObjectSize=int(SURFACE_MIN_MM2 /(sx*sy*sz)),
        sortByObjectSize=True
    )

    if afficher_intermediaire:
        from ..utils.visualization import afficher_grille
        afficher_grille(
            [composantes, composantes_tries],
            ['Composantes brutes', 'Composantes filtrées'],
            no_window=True
        )

    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    shape_stats.Execute(composantes_tries)
    if not silent:
        logger.info(f"Régions candidates : {shape_stats.GetNumberOfLabels()}")

    # ── Préparer le gradient une seule fois (coûteux) ─────────────────────
    img_closed  = sitk.ClosingByReconstruction(img, [2, 2, 0])
    img_gradient = sitk.GradientMagnitude(img_closed)

    # ── Étape 4 : filtrage morphologique + validation ─────────────────────
    
    
    for label in shape_stats.GetLabels():
        surface = shape_stats.GetPhysicalSize(label)
        elong   = shape_stats.GetElongation(label)
        bbox    = shape_stats.GetBoundingBox(label)
        

        # Filtre 1 : surface et élongation
        if not (SURFACE_MIN_MM2 <= surface <= SURFACE_MAX_MM2
                and ELONG_MIN <= elong <= ELONG_MAX):
            continue
        if not silent:
            logger.debug(f"Label {label} : surface={surface:.0f} mm², elong={elong:.2f}")

        # Centroïde via médiane (plus robuste que le barycentre)
        arr_comp = sitk.GetArrayFromImage(composantes_tries == label)
        centroid = np.median(np.argwhere(arr_comp), axis=0)[::-1]
        seed     = [int(c) for c in centroid]

        # Étape 5 : validation par ConfidenceConnected sur le gradient
        seg = sitk.ConfidenceConnected(
            img_gradient,
            seedList=[seed],
            numberOfIterations=3,
            multiplier=2.5,
            initialNeighborhoodRadius=2,
            replaceValue=1,
        )

        shape_filter = sitk.LabelShapeStatisticsImageFilter()
        shape_filter.Execute(seg)

        # Filtre 2 : surface totale après ConfidenceConnected
        surface_totale = sum(
            shape_filter.GetPhysicalSize(l)
            for l in shape_filter.GetLabels()
        )
        if surface_totale > SURFACE_MAX_MM2:
            if not silent:
                logger.debug(f"Label {label} rejeté : surface ConfConn trop grande ({surface_totale:.0f})")
            continue

        # Filtre 3 : largeur moyenne via RLE
        # ⚠️ GetRLEIndexes(1) crash si label 1 absent → vérification préalable
        if 1 not in shape_filter.GetLabels():
            if not silent:
                logger.debug(f"Label {label} rejeté : ConfidenceConnected vide")
            continue

        rle_indexes  = shape_filter.GetRLEIndexes(1)
        # Format RLE : (slice, row, start_col, length, ...)
        # [3::4] extrait les longueurs de runs = largeurs horizontales
        largeurs     = np.array(rle_indexes[3::4])
        mean_largeur = largeurs.mean() if len(largeurs) > 0 else 0

        if mean_largeur < LARGEUR_MIN_PX:
            if not silent:
                logger.debug(f"Label {label} rejeté : largeur trop faible ({mean_largeur:.1f}px)")
            continue
        
        mask_lateralite = calculer_lateralite([bbox], longueur_x, est_profil)
        
        if not mask_lateralite:
            if not silent:
                logger.debug(f"Label {label} rejeté : chevauche la ligne médiane")
            continue

        # ── Candidat validé ──────────────────────────────────────────────
        mask          = sitk.Cast(seg, sitk.sitkUInt8) \
                      | sitk.Cast(composantes_tries == label, sitk.sitkUInt8)
        masque_final  = masque_final | mask
        bboxes.append(bbox)
        prothese      = True
        
        if not silent:
            logger.info(f"Prothèse détectée — label {label}, surface {surface:.0f} mm², elong {elong:.2f}")

        if afficher_intermediaire:
            from ..utils.visualization import afficher_grille
            afficher_grille(
                [composantes_tries == label],
                [f"Label {label} | surface {surface:.0f} mm² | élongation {elong:.2f}"],
                no_window=True
            )

    lateralite = calculer_lateralite(bboxes, longueur_x, est_profil)
    
    return prothese, masque_final, bboxes, lateralite