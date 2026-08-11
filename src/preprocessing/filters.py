# ─────────────────────────────────────────────
# src/preprocessing/filters.py
# ─────────────────────────────────────────────
import SimpleITK as sitk
import numpy as np
import scipy.ndimage as ndi


def nettoyage_image_rapide(image: sitk.Image, seuil_voisins: int = 10) -> sitk.Image:
    """
    Supprime les pixels isolés d'une image binaire par convolution vectorisée.

    Un pixel à 1 est supprimé si le nombre de ses voisins actifs
    est inférieur à `seuil_voisins`.

    Args:
        image:          Image binaire SimpleITK (valeurs 0/1).
        seuil_voisins:  Nombre minimum de voisins pour conserver un pixel.

    Returns:
        Image binaire nettoyée, avec les mêmes métadonnées spatiales.
    """
    arr = sitk.GetArrayFromImage(image).astype(np.uint8)

    # Kernel 5x5 (2D) ou 5x5x5
    kernel = np.ones((5,) * arr.ndim, dtype=np.uint8)
    kernel[tuple(1 for _ in range(arr.ndim))] = 0  # Exclure le pixel central

    # Compte le nombre de voisins à 1 pour chaque pixel
    nb_voisins = ndi.convolve(arr, kernel, mode='constant', cval=0)

    # Suppression des pixels trop isolés
    arr_clean = arr.copy()
    arr_clean[(arr == 1) & (nb_voisins < seuil_voisins)] = 0

    img_clean = sitk.GetImageFromArray(arr_clean)
    img_clean.CopyInformation(image)
    return img_clean


def seuil_par_percentile(image_sitk: sitk.Image, percentile: float) -> float:
    """
    Calcule la valeur d'intensité correspondant à un percentile donné.

    Utile pour définir un seuil robuste au bruit et aux valeurs extrêmes,
    en concentrant la dynamique sur les structures denses (os, métal).

    Args:
        image_sitk: Image SimpleITK (toute dimension).
        percentile: Percentile voulu [0, 100].
                    Exemple : 97.5 → seul le 2.5 % supérieur est gardé.

    Returns:
        Valeur HU (float) correspondant au percentile.
    """
    arr = sitk.GetArrayFromImage(image_sitk)
    return float(np.percentile(arr, percentile))