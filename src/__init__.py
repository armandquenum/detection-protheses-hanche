"""
src — Détection et localisation 3D de prothèses de hanche.

Pipeline : détection/segmentation 2D sur topogrammes CT (SimpleITK +
nnU-Net) -> localisation 3D par reconstruction multi-angles -> (à venir)
calcul SUV et corrélation clinique. Voir ARCHITECTURE.md à la racine du
dépôt pour le détail du pipeline et des formats de données.

Sous-packages :
    annotation       — génération et validation des annotations (Napari)
    detection        — segmentation SimpleITK (topogrammes 2D)
    segmentation     — inférence nnU-Net
    evaluation       — métriques et comparaison des méthodes
    preprocessing    — chargement DICOM->NIfTI, filtres, pipeline de prétraitement
    localisation3d   — BBox 3D à partir des vues coronale/sagittale
    reconstruction3d — projection multi-angles et reconstruction 3D
    suv              — (à venir) calcul du SUV péri-prothétique
    clinical         — (à venir) corrélation avec les données cliniques
    utils            — I/O, chemins, visualisation — utilisé par tous les
                        autres sous-packages

Ce fichier ne réexporte volontairement rien au niveau du package racine :
tout le projet importe explicitement depuis le sous-module concerné
(ex. `from src.utils.io import resoudre_dicom_path`), jamais depuis
`src` directement. Gardez cette convention dans les __init__.py des
sous-packages : un __init__.py vide (ou avec seulement une docstring
décrivant le rôle du sous-package, sur ce modèle) plutôt que des
réexports, pour éviter les imports circulaires entre modules qui se
référencent déjà mutuellement (ex. reconstruction3d <-> evaluation).
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("stage-m1-prosthesis-detection")
except PackageNotFoundError:
    # Paquet non installé (ex. exécution directe depuis le repo sans
    # `pip install -e .`) — valeur de repli, pas une vraie version.
    __version__ = "0.0.0-dev"