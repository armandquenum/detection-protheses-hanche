# ─────────────────────────────────────────────
# src/utils/io.py
# ─────────────────────────────────────────────
from __future__ import annotations
import logging
import json
import SimpleITK as sitk
import numpy as np
from datetime import datetime as dt
from pathlib import Path
import pydicom
from pydicom.dataset import Dataset
from typing import TYPE_CHECKING

if TYPE_CHECKING:
       from pydicom.multival import MultiValue
 
from src.utils.paths import DATA
 
logger = logging.getLogger(__name__)

 
 
# ─────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────
 
# Codes DICOM des métadonnées à extraire pour chaque patient.
# Clé = nom lisible utilisé dans les JSON d'annotations.
# Valeur = tag DICOM au format "groupe|élément".
METADATAS = {
    "jour_d_examen":                        [(0x0008, 0x0022)],
    "date_de_naissance":                    [(0x0010, 0x0030)],
    "requested_procedure_description":      [(0x0032, 0x1060)],
    "scheduled_procedure_step_description": [(0x0040, 0x0275), (0x0040, 0x0007)],
}
 
 
# ─────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────────────────────
 
def charger_dicom_paths(data_folder: str | Path=DATA) -> np.ndarray:
    """
    Charge la liste des chemins DICOM sauvegardée par le pipeline de conversion.
 
    Le fichier dicom_paths.npy est généré par loader.run_conversion_pipeline().
    S'il est absent (première installation, environnement CI, etc.),
    retourne un tableau vide plutôt que de crasher à l'import.
 
    Returns:
        Array numpy de chemins (str), vide si le fichier est introuvable.
    """
    path = Path(data_folder) / 'dicom_paths.npy'
    if not path.exists():
        logger.warning("dicom_paths.npy introuvable — métadonnées DICOM indisponibles")
        return np.array([])
    return np.load(path)
 
 
def charger_json(json_path: str | Path) -> dict:
    """
    Charge un fichier JSON et retourne son contenu.
 
    Args:
        json_path: Chemin vers le fichier JSON.
 
    Returns:
        Dictionnaire chargé, ou dict vide si le fichier est absent.
    """
    path = Path(json_path)
    if path.exists():
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return {}
 
 
def sauvegarder_json(dictionnaire: dict, json_path: str | Path) -> None:
    """
    Sauvegarde un dictionnaire dans un fichier JSON (UTF-8, indenté).
 
    Args:
        dictionnaire: Données à sérialiser.
        json_path:    Chemin de destination (créé s'il n'existe pas).
    """
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dictionnaire, f, indent=2, ensure_ascii=False)


def dossier_predictions(model: str, dataset_path: str | Path, version: int = 1) -> Path:
    """
    Point d'entrée unique pour le mapping modèle → dossier de sortie des
    prédictions en mode eval. Remplace la logique dupliquée entre
    predictor.py ('nnunet_seg'/'nnunetv2_seg') et pipeline_annotation.py
    ('sitk_seg', codé en dur) — un seul endroit à modifier si les noms
    de dossiers changent un jour.

    Args:
        model:        'nnunet', 'nnunetv2', ou 'sitk'.
        dataset_path: Chemin du dataset DatasetXXX_Nom.
        version:      1 ou 2 — utilisé seulement si model='nnunet', pour
                      distinguer nnunet_seg/nnunetv2_seg (redondant avec
                      model='nnunetv2', gardé pour compat avec predictor.py).

    Returns:
        Chemin du dossier (non créé) où lire/écrire segmentations.json.

    Raises:
        ValueError: si model n'est pas reconnu.
    """
    dataset_path = Path(dataset_path)

    if model == 'sitk':
        return dataset_path / 'sitk_seg'
    if model in ('nnunet', 'nnunetv2'):
        v = 2 if model == 'nnunetv2' else version
        return dataset_path / ('nnunetv2_seg' if v == 2 else 'nnunet_seg')
    raise ValueError(f"model '{model}' non reconnu — attendu 'sitk', 'nnunet' ou 'nnunetv2'")
 
 
# ─────────────────────────────────────────────────────────────
# MÉTADONNÉES DICOM
# ─────────────────────────────────────────────────────────────
 
def lire_metadata_dicom(dicom_path: str) -> dict[str, str | None]:
    """
    Extrait les métadonnées cliniques d'une image DICOM.

    Les champs listés dans METADATAS sont extraits si présents.
    Les dates (format YYYYMMDD) sont converties en 'YYYY/MM/DD'.

    Args:
        dicom_path: Chemin vers le fichier DICOM.

    Returns:
        Dictionnaire {nom_lisible: valeur_str | None}.
        Une clé vaut None si le tag est absent de l'image.
    """
    ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
    resultats: dict[str, str | None] = {}

    for name, path_tags in METADATAS.items():
        resultats[name] = _extraire_valeur(ds, path_tags)

    return resultats


def _extraire_valeur(ds: Dataset, path_tags: list[tuple[int, int]]) -> str | None:
    """Navigue dans ds (éventuellement à travers des séquences) pour extraire un champ."""
    container = ds  # Dataset courant (racine ou item de séquence)
    element = None  # DataElement final trouvé

    for i, tag in enumerate(path_tags):
        if tag not in container:
            return None
        element = container[tag]

        # S'il reste des tags à parcourir, on descend dans la séquence
        if i < len(path_tags) - 1:
            if not element.value:  # séquence vide
                return None
            container = element.value[0]  # premier item de la séquence

    if element is None:
        return None

    if element.VR == "DA":
        try:
            return dt.strptime(element.value, "%Y%m%d").date().strftime("%Y/%m/%d")
        except (ValueError, TypeError):
            return None

    return str(element.value) if element.value is not None else None
 
 
# ─────────────────────────────────────────────────────────────
# GÉOMÉTRIE DES MASQUES
# ─────────────────────────────────────────────────────────────

def detecter_vue_profil_dicom(dicom_path: str) -> dict:
    """
    Détecte si un topogramme est une vue de profil
    depuis les métadonnées DICOM.
    
    Returns:
        dict avec 'est_profil' (bool) et 'methode' (str)
    """
    try:
        ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
        # stop_before_pixels=True → lecture rapide, pas besoin des pixels
    except:
        return {'est_profil': None, 'methode': 'indéterminé'}

    # Méthode 1 — ViewPosition (le plus explicite)
    if hasattr(ds, 'ViewPosition'):
        vp = ds.ViewPosition.strip().upper()
        if vp in ('LAT', 'LL', 'RL', 'LLAT', 'RLAT'):
            return {'est_profil': True,  'methode': 'ViewPosition=' + vp}
        if vp == 'AP':
            return {'est_profil': False, 'methode': 'ViewPosition=AP'}

    # Méthode 2 — SeriesDescription (Siemens souvent "Topogram 90°")
    if hasattr(ds, 'SeriesDescription'):
        desc = ds.SeriesDescription.strip().upper()
        if '90' in desc or 'LAT' in desc or 'PROFIL' in desc:
            return {'est_profil': True,  'methode': 'SeriesDescription=' + desc}
        if '0°' in desc or 'AP' in desc or 'FACE' in desc:
            return {'est_profil': False, 'methode': 'SeriesDescription=' + desc}

    # Méthode 3 — PatientOrientation (0020,0020)
    if hasattr(ds, 'PatientOrientation'):
        orient = ds.PatientOrientation
        if isinstance(orient, (list, MultiValue)):
            orient_str = '\\'.join(str(o).upper() for o in orient)
        else:
            orient_str = str(orient).upper()
        
        # Vue profil → lignes dans axe A/P
        if orient_str in ('A\\P', 'P\\A', 'A\\F', 'F\\A'):
            return {'est_profil': True,  'methode': 'PatientOrientation=' + orient_str}
        # Vue face → lignes dans axe L/R
        if orient_str in ('L\\P', 'R\\P', 'L\\F', 'R\\F'):
            return {'est_profil': False, 'methode': 'PatientOrientation=' + orient_str}

    # Méthode 4 — ImageOrientation (0020,0037) si disponible
    if hasattr(ds, 'ImageOrientationPatient'):
        iop = [float(x) for x in ds.ImageOrientationPatient]
        # Vecteur ligne = [iop[0], iop[1], iop[2]]
        # Vue AP  → vecteur ligne ≈ [1,0,0] (axe X = gauche/droite)
        # Vue LAT → vecteur ligne ≈ [0,1,0] (axe Y = avant/arrière)
        row_vec = iop[:3]
        if abs(row_vec[1]) > 0.9:  # composante Y dominante → profil
            return {'est_profil': True,  'methode': 'ImageOrientationPatient'}
        if abs(row_vec[0]) > 0.9:  # composante X dominante → face
            return {'est_profil': False, 'methode': 'ImageOrientationPatient'}

    return {'est_profil': None, 'methode': 'indéterminé'}


def calculer_lateralite(bboxes: list, longueur_x: int, est_profil: bool | None) -> int:
    """
    Détermine la latéralité de l'ensemble des prothèses détectées.
 
    Convention de retour :
        -1 → orientation indéterminée (est_profil inconnu)
        0  → aucune prothèse (aucune bbox ne passe les critères)
        1  → prothèse à gauche uniquement
        2  → prothèse à droite uniquement
        3  → prothèses bilatérales (gauche + droite)
        4  → vue de profil et prothèse latéralité indéterminé
 +
    Une bbox est ignorée si elle chevauche la ligne médiane
    (ni entièrement à gauche, ni entièrement à droite).
 
    La dimension de la bbox est déduite de sa longueur :
        longueur 4 → 2D : (x, y, w, h)       → largeur = bbox[2]
        longueur 6 → 3D : (x, y, z, w, h, d) → largeur = bbox[3]
 
    Args:
        bboxes:    Liste de bounding boxes au format GetBoundingBox() de SimpleITK.
        longueur_x: Taille de l'image sur l'axe X (nombre de pixels/voxels).
        est_profil: Booléen indiquant si l'image est en vue de profil (None si indéterminé)

 
    Returns:
        Entier dans {-1, 0, 1, 2, 3, 4} selon la latéralité détectée.
    """

    # Cas où rien n'a été segmenter
    if not bboxes:
        return 0
    # Cas où on ne siat pas si on a une vue de profil ou non (cas indéterminé)

    if est_profil == None:
        return -1

     # Cas vue de profil
    if est_profil:
        return 4  # ne pas calculer G/D sur cette bbox

    lateralite_gauche  = False
    lateralite_droite  = False

    for bbox in bboxes:
        idx_largeur = 3 if len(bbox) == 6 else 2
        x_debut = bbox[0]
        x_fin   = bbox[0] + bbox[idx_largeur]
        milieu  = longueur_x / 2

        # Entièrement à gauche
        if x_fin <= milieu:
            lateralite_gauche = True

        # Entièrement à droite
        elif x_debut >= milieu:
            lateralite_droite = True

    # Résultat

    if lateralite_gauche and lateralite_droite:
        return 3  # bilatérale

    if lateralite_gauche:
        return 1

    if lateralite_droite:
        return 2

    return 0
 
 
def extraire_bboxes(masque: sitk.Image, surface_min_mm2: float = 2.0) -> list:
    """
    Extrait les bounding boxes des composantes connexes d'un masque binaire.
 
    Pipeline :
        1. ConnectedComponent       → labellise les régions connexes
        2. RelabelComponent         → filtre les petites régions (< surface_min_mm2)
        3. LabelShapeStatistics     → calcule les propriétés géométriques
        4. GetBoundingBox           → extrait les bboxes pour chaque label
 
    Args:
        masque:          Masque binaire SimpleITK (valeurs 0/1).
        surface_min_mm2: Surface minimale en mm² pour conserver une composante.
 
    Returns:
        Liste de bounding boxes au format GetBoundingBox() de SimpleITK.
        Vide si le masque ne contient aucune composante valide.
    """
    surface_scale = np.array(masque.GetSpacing()).prod()
 
    composantes = sitk.ConnectedComponent(sitk.Cast(masque, sitk.sitkUInt8))
    composantes_tries = sitk.RelabelComponent(
        composantes,
        minimumObjectSize=int(surface_min_mm2 / surface_scale),
        sortByObjectSize=True,
    )
 
    shape = sitk.LabelShapeStatisticsImageFilter()
    shape.Execute(composantes_tries)
 
    return [shape.GetBoundingBox(label) for label in shape.GetLabels()]


# ─────────────────────────────────────────────────────────────
# RÉSOLUTION DICOM
# ─────────────────────────────────────────────────────────────

def resoudre_dicom_path(patient_id: str, dicom_paths: np.ndarray) -> str:
    """
    Recherche un chemin DICOM par sous-chaîne sur un ID brut (pré-nnU-Net).

    Utilisé uniquement pour les ID bruts (ex: 'ACT1.cem.1fcf.fr...'),
    jamais pour un case_id nnU-Net (voir resoudre_dicom_gt).

    Args:
        patient_id:  ID brut (nom de fichier NIfTI sans suffixe).
        dicom_paths: Array de chemins DICOM (charger_dicom_paths()).

    Returns:
        Chemin DICOM trouvé, ou '' si aucun match.
    """
    try:
        return dicom_paths[np.strings.find(dicom_paths, patient_id) > 0][0]
    except IndexError:
        return ''