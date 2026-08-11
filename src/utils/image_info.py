# ─────────────────────────────────────────────
# src/utils/image_info.py
# ─────────────────────────────────────────────
import SimpleITK as sitk
import numpy as np

def get_descript_img(image: sitk.Image) -> None:
    """Affiche la taille, le spacing et la plage d'intensité d'une image."""
    arr = sitk.GetArrayFromImage(image)
    print(f"  Taille   : {image.GetSize()}")
    print(f"  Spacing  : {tuple(f'{s:.2f}' for s in image.GetSpacing())} mm")
    print(f"  Plage HU : [{arr.min():.0f}, {arr.max():.0f}]")
    print(f"  Type     : {image.GetPixelIDTypeAsString()}")