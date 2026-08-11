# ─────────────────────────────────────────────
# src/preprocessing/pipeline.py
# ─────────────────────────────────────────────
import logging
import SimpleITK as sitk
import numpy as np
from src.utils.image_info import get_descript_img

logger = logging.getLogger(__name__)


def preprocessing_image(img: sitk.Image, silent: bool = True) -> sitk.Image:
    """
    Prépare une image topogramme pour la segmentation.

    Pipeline :
        1. Cast Float32
        2. Clamp percentile [1, 99] → supprime les valeurs aberrantes

    Note : pas de normalisation (centrage/réduction) — le seuillage automatique
    (seuil_percent, voir get_segmentation) donne de meilleurs résultats sur les
    valeurs brutes ; normaliser rapprocherait les valeurs et déplacerait le
    seuil optimal.
    Pas de spacing forcé — certains topogrammes sont recadrés/zoomés (ex:
    512×512×1 centré sur le corps) ; les mesures physiques réelles (mm)
    restent donc plus comparables entre patients que des mesures en pixels.

    Args:
        img: Image SimpleITK brute (topogramme).
        silent: Ne pas faire d'affichages

    Returns:
        Image normalisée (Float32, clampée), prête pour la segmentation.
    """
    
    if not silent:
        logger.debug("── Preprocessing ──")
        get_descript_img(img)

    # 1. Conversion Float32
    image_f = sitk.Cast(img, sitk.sitkFloat32)

    # 2. Clamp sur les percentiles [1, 99] pour ignorer les artefacts extrêmes
    arr    = sitk.GetArrayFromImage(image_f)
    p_low  = float(np.percentile(arr,  1))
    p_high = float(np.percentile(arr, 99))
    image_clamp = sitk.Clamp(image_f, lowerBound=p_low, upperBound=p_high)

    
    if not silent:
        logger.debug("── Après preprocessing ──")
        get_descript_img(image_clamp)
    return image_clamp