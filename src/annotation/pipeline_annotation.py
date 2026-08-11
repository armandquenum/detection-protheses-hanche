# src/annotation/pipeline_annotation.py

from __future__ import annotations
import logging
import re
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari


from src.detection.segmentation import get_segmentation
from src.preprocessing.pipeline import preprocessing_image
from src.utils.io import (
    charger_dicom_paths,
    charger_json,
    sauvegarder_json,
    lire_metadata_dicom,
    calculer_lateralite,
    extraire_bboxes,
    detecter_vue_profil_dicom,
    resoudre_dicom_path,
    dossier_predictions,
)

logger = logging.getLogger(__name__)

LABEL_LAYER_NAME = "masque"


from src.detection.constants import SURFACE_MIN_MM2



# ─────────────────────────────────────────────────────────────
# 1. GÉNÉRATION AUTOMATIQUE DES MASQUES CANDIDATS
# ─────────────────────────────────────────────────────────────

def generer_annotations_candidates(
    nifti_paths: list[str],
    output_dir: str,
    dataset_path: str='',
    mode: str='annotation',
    merge_into_annotations: bool = True,
) -> dict:
    """
    Passe 1 — Génère automatiquement les masques candidats via SimpleITK.
    Produit les fichiers à corriger dans l'outil d'annotation.

    NB: cette fonction est réutiliser pour comparer la segmentation SimpleITK et nnU-Net.

        Le paramètre mode sélectionne le comportement : :
            - Constitution initiale du dataset: mode ='annotation'.
                Structure de sortie :
                    output_dir/
                    ├── masks_auto/       ← masques générés par SimpleITK
                    ├── masks_corrected/  ← masques corrigés manuellement
                    └── annotations.json  ← fichier de suivi du statut
            - Évaluation sur le dataset nnU-Net : mode ='eval', il faut founir absolument le json de la vérité terrain gt_json.
                

    Args:
        nifti_paths:            Liste des chemins NIfTI (mode='annotation' uniquement).
        output_dir:             Dossier de travail pour l'annotation (mode='annotation').
        dataset_path:           Chemin du dataset nnU-Net Dataset0XX_Nom (mode='eval'),
                                doit contenir gt.json (généré par exporter_nnunet).
        mode:                   Mode dans lequel est lancée la fonction
                                    annotation: pour génerer l'annotation dans la phase de développemet
                                    eval:       pour évaluer la segmentation par SimpleITK utile pour la comparaison avec nnunet
        merge_into_annotations : bool, default True
            Contrôle si les masques candidats générés sont fusionnés dans le
            fichier `annotations.json` global (celui utilisé ensuite par
            `exporter_nnunet()` pour construire le dataset nnU-Net — voir
            ARCHITECTURE.md §1 étape [2]), avec le schéma complet du workflow
            d'annotation (`prothese_auto`, `lateralite_auto`, `mask_auto`,
            `mask_corrected`, `statut`).

            - True (comportement historique) : à réserver strictement à
              l'étape d'annotation du pipeline (mode='annotation' appelé
              depuis le flux de préparation du dataset, avant validation
              manuelle dans Napari).
            - False : aucune écriture dans `annotations.json`. Les résultats
              sont écrits dans un `segmentations.json` dans `output_dir`,
              avec le même schéma de champs que les autres chemins de
              l'étape [5] (`prothese`, `lateralite`, `mask`, `bboxes_list`,
              `est_profil`) — pas de `statut`/`mask_corrected`, qui n'ont de
              sens que pour un workflow de correction manuelle. À utiliser
              pour tout appel ad-hoc/exploratoire (ex.
              `scripts/run_inference.py -m sitk`).

    Returns:
        Dictionnaire de statut par patient.
    """
    
    

    

    if mode not in ('annotation', 'eval'):
        logger.warning(f"mode: {mode} non reconnu, choisir entre ('annotation'/'eval')")
        return {}

    if mode == 'annotation':
        out = Path(output_dir)
        subs = ['masks_auto', 'masks_corrected']
        for sub in subs:
            (out / sub).mkdir(parents=True, exist_ok=True)

        annotations_path = out / ('annotations.json' if merge_into_annotations else 'segmentations.json')
        annotations      = charger_json(annotations_path)
        deja_traites     = {pid for pid, v in annotations.items() if v.get('statut') != None}
        dicom_paths = charger_dicom_paths()
        for idx, nifti_path in enumerate(tqdm(nifti_paths, desc="Génération masques", unit="patient")):
            patient_id = _extraire_patient_id(nifti_path)

            # Skip si déjà traité (reprise après interruption)
            if patient_id in deja_traites:
                logger.debug(f"Skip {patient_id} — déjà traité")
                continue

            try:
                img     = sitk.ReadImage(nifti_path)
                img_pre = preprocessing_image(img)
                if img.GetDimension() == 2:
                    img_pre = sitk.JoinSeries([img_pre])
                dicom_path = resoudre_dicom_path(patient_id, dicom_paths)
                est_profil = detecter_vue_profil_dicom(dicom_path)['est_profil']
                
                if est_profil == None:
                    logger.warning(f"Orientation de l'image inconnu erreur possible sur la segmentation de {patient_id}")
                prothese, masque, bboxes, lateralite = get_segmentation(img_pre, est_profil)

                mask_out = out / 'masks_auto' / f"{patient_id}.nii.gz"

                if img.GetDimension() == 2:
                    size = list(masque.GetSize())
                    size[2] = 0
                    masque = sitk.Extract(masque,size,[0,0,0])
                

                sitk.WriteImage(masque, str(mask_out))
                if merge_into_annotations:
                    annotations[patient_id] = {
                        'image':            str(nifti_path),
                        'mask_auto':        str(mask_out),
                        'mask_corrected':   str(out / 'masks_corrected' / f"{patient_id}.nii.gz"),
                        'prothese_auto':    prothese,
                        'lateralite_auto':  lateralite,
                        'bboxes_list':      bboxes,
                        'statut':           'à_valider',
                        'est_profil':       est_profil,
                    }
                else:
                    # Schéma aligné sur segmentations.json (ARCHITECTURE.md §3) —
                    # pas de champs 'statut'/'mask_corrected', qui n'ont de sens
                    # que pour le workflow d'annotation avec correction Napari.
                    annotations[patient_id] = {
                        'image':       str(nifti_path),
                        'mask':        str(mask_out),
                        'prothese':    prothese,
                        'lateralite':  lateralite,
                        'bboxes_list': bboxes,
                        'est_profil':  est_profil,
                    }
                    
                if dicom_path:
                    annotations[patient_id].update(lire_metadata_dicom(dicom_path))
                

            except Exception as e:
                logger.error(f"Erreur sur {patient_id} : {e}")
                annotations[patient_id] = {'statut': 'erreur', 'message': str(e)}

            # Sauvegarde intermédiaire toutes les 10 images (sécurité)
            if idx % 10 == 0:
                sauvegarder_json(annotations, annotations_path)
        
        sauvegarder_json(annotations, annotations_path)
        _afficher_resume(annotations, annotations_path)
        return annotations
    
    else:
        path_of_dataset = Path(dataset_path)
        dataset_name = path_of_dataset.name
        is_dataset_name = re.search(r"^Dataset\d{3}_[a-zA-Z0-9_]+$", dataset_name)
        
        if path_of_dataset.exists() and is_dataset_name:
            out = dossier_predictions('sitk', path_of_dataset)
            gt_json = charger_json(path_of_dataset /'gt.json')
            if not out.exists():
                out.mkdir(parents=True, exist_ok=True)
            
            annotations = charger_json(out / 'segmentations.json')

            deja_traites     = {case_id for case_id, v in annotations.items() if Path(v.get('mask')).exists()}

            for idx, case_id in enumerate(tqdm(gt_json.keys(), desc="Génération masques", unit="patient")):
                
                if case_id in deja_traites:
                    logger.debug(f"Skip {case_id} — déjà traité")
                    continue
                try:
                    info = gt_json[case_id]
                    img     = sitk.ReadImage(info['image'])
                    img_pre = preprocessing_image(img)
                    if img.GetDimension() == 2:
                        img_pre = sitk.JoinSeries([img_pre])

                    est_profil = info.get('est_profil')

            
                    prothese, masque, bboxes, lateralite = get_segmentation(img_pre, est_profil)

                    mask_out = out / f"{case_id}.nii.gz"

                    if img.GetDimension() == 2:
                        size = list(masque.GetSize())
                        size[2] = 0
                        masque = sitk.Extract(masque,size,[0,0,0])
                    

                    sitk.WriteImage(masque, str(mask_out)) 
                    annotations[case_id] = {
                        'image':       str(info['image']),
                        'mask':        str(mask_out),
                        'prothese':    prothese,
                        'lateralite':  lateralite,
                        'est_profil':       est_profil
                    }
                except Exception as e:
                    logger.error(f"Erreur sur {case_id} : {e}")
                    annotations[case_id] = {'statut': 'erreur', 'message': str(e)}

                # Sauvegarde intermédiaire toutes les 10 images (sécurité)
                if idx % 10 == 0:
                    sauvegarder_json(annotations, out / 'segmentations.json')

            sauvegarder_json(annotations, out / 'segmentations.json')
            _afficher_resume(annotations, out / 'segmentations.json')
            return annotations
        else:
            logger.warning(f"{dataset_path} : dossier introuvable ou nom invalide (attendu 'DatasetXXX_Nom')")
            return {}

# ─────────────────────────────────────────────────────────────
# 2. ANNOTATION MANUELLE AVEC NAPARI (SÉQUENTIELLE)
# ─────────────────────────────────────────────────────────────
def lancer_annotation_napari(
    image: str | sitk.Image,
    masque: str | sitk.Image | None = None,
) -> napari.Viewer:
    """
    Ouvre une fenêtre Napari avec l'image et le masque candidat.
    Retourne le viewer sans démarrer l'event loop (appeler napari.run() ensuite).

    Args:
        image:  Chemin NIfTI ou image SimpleITK.
        masque: Chemin NIfTI, image SimpleITK, ou None (masque vide).

    Returns:
        Viewer Napari configuré (event loop non démarré).
    """
    import napari  # Import lazy — napari est lourd

    arr = _sitk_to_array(image)

    viewer = napari.Viewer(title="Annotation prothèse")
    viewer.add_image(arr, name='topogramme', colormap='gray')

    # Masque : existant ou vide
    arr_m = _sitk_to_array(masque) if masque else np.zeros(arr.shape if arr else 0, dtype=np.uint8)
    viewer.add_labels(arr_m.astype(np.uint8) if arr_m else np.array([]), name=LABEL_LAYER_NAME)

    # Instructions affichées dans le titre
    viewer.title = "Annotation | Corriger le masque puis FERMER la fenêtre | S = sauvegarder"

    return viewer


def validation_annotations_candidates(
    annotations_json_path: str,
    positifs_seulement: bool = True,
    statuts_cibles: tuple[str, ...] = ('à_valider',),
) -> dict:
    """
    Lance une session d'annotation séquentielle dans UN SEUL viewer Napari.

    Workflow : parcourir les patients ciblés (statuts_cibles), corriger le
    masque directement dans le layer 'masque' du viewer, et sauvegarder.
    Les touches F/P servent spécifiquement à corriger l'orientation
    (est_profil) quand la détection automatique depuis les métadonnées
    DICOM (detecter_vue_profil_dicom) s'est trompée ou n'a rien détecté
    (voir champ 'est_profil' : True/False/None) — la correction est
    sauvegardée immédiatement.

    Raccourcis clavier :
        S           = sauvegarder le patient courant (masque + orientation)
        RIGHT_ARROW = passer au patient suivant
        LEFT_ARROW  = passer au patient précédent
        F           = marquer la vue courante comme FACE, sauvegarder
        P           = marquer la vue courante comme PROFIL, sauvegarder
        Q           = quitter la session (propose de sauvegarder avant)

    Args:
        annotations_json_path: Chemin vers annotations.json.
        positifs_seulement:    Si True, ne montre que les détections positives.
        statuts_cibles:        Statuts à traiter (ex: ('à_valider',)).

    Returns:
        Dictionnaire d'annotations mis à jour (mêmes objets que le JSON
        sur disque, déjà sauvegardés au fil de la session).
    """
    import napari

    json_path   = Path(annotations_json_path)
    annotations = charger_json(json_path)

    cibles = {
        pid: info for pid, info in annotations.items()
        if info.get('statut') in statuts_cibles
        and (not positifs_seulement or info.get('prothese', False))
    }

    if not cibles:
        print("Aucun patient à annoter avec ces critères.")
        return annotations

    patients = list(cibles.items())
    total    = len(patients)
    print(f"\n{total} patients à annoter")
    print("S = sauvegarder | → = suivant | ← = précédent | F = vue de face | P = vue de profil | Q = quitter\n")

    idx = [0]

    def _construire_titre(i: int, pid: str, info: dict) -> str:
        """Construit le titre de la fenêtre — factorisé pour être réutilisé
        sans recharger l'image (voir rafraichir_titre)."""
        est_profil = info.get('est_profil')
        orientation = "PROFIL" if est_profil is True else "FACE" if est_profil is False else "INDÉTERMINÉ"
        return (
            f"[{i+1}/{total}] {pid}  |  "
            f"Prothèse auto : {'OUI' if info.get('prothese_auto') else 'NON'}  |  "
            f"Orientation : {orientation}  |  "
            f"S=sauvegarder  →=suivant  ←=précédent  F=face  P=profil  Q=quitter"
        )

    def charger_patient(viewer, i: int):
        """Met à jour les layers ET le titre — utilisé au changement de patient."""
        pid, info = patients[i]
        arr = sitk.GetArrayFromImage(sitk.ReadImage(info['image']))

        mask_path = info.get('mask_auto', '') if info.get('statut') != "validé" else info.get('mask_corrected', '')
        if mask_path and Path(mask_path).exists():
            arr_m = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        else:
            arr_m = np.zeros(arr.shape, dtype=np.uint8)

        viewer.layers['topogramme'].data = arr
        viewer.layers[LABEL_LAYER_NAME].data = arr_m.astype(np.uint8)
        viewer.reset_view()
        viewer.title = _construire_titre(i, pid, info)
        print(f"[{i+1}/{total}] {pid}")

    def rafraichir_titre(viewer, i: int):
        """Met à jour SEULEMENT le titre — pas de relecture disque, pas de
        reset du zoom. Utilisé après une correction F/P (l'image et le
        masque n'ont pas changé, seule l'orientation a changé)."""
        pid, info = patients[i]
        viewer.title = _construire_titre(i, pid, info)

    def sauvegarder_courant(viewer):
        pid, info = patients[idx[0]]
        img_ref   = sitk.ReadImage(info['image'])
        if _sauvegarder_masque_corrige(
            viewer, pid, info, img_ref, annotations, json_path
        ):
            print(f"  💾 Sauvegardé — {pid}")
        else:
            print(f"  💾 Non sauvegardé — {pid} : problème rencontré")

    # ── Créer le viewer UNE SEULE FOIS ────────────────────────────────────
    pid0, info0 = patients[0]
    arr0  = sitk.GetArrayFromImage(sitk.ReadImage(info0['image']))
    mask_path0 = info0.get('mask_auto', '') if info0.get('statut') != "validé" else info0.get('mask_corrected', '')
    arr_m0 = (
        sitk.GetArrayFromImage(sitk.ReadImage(mask_path0)).astype(np.uint8)
        if mask_path0 and Path(mask_path0).exists()
        else np.zeros(arr0.shape, dtype=np.uint8)
    )
    
    viewer = napari.Viewer()
    viewer.add_image(arr0, name='topogramme', colormap='gray')
    viewer.add_labels(arr_m0, name=LABEL_LAYER_NAME)
    viewer.title = _construire_titre(0, pid0, info0)
    
    # ── Raccourcis clavier ────────────────────────────────────────────────
    @viewer.bind_key('s', overwrite=True)
    def on_save(v):
        """S — sauvegarder sans changer de patient."""
        sauvegarder_courant(v)
    
    @viewer.bind_key('Right', overwrite=True)
    def on_next(v):
        """→ — passer au patient suivant."""
        idx[0] += 1
        if idx[0] < total:
            charger_patient(v, idx[0])
        else:
            idx[0] -= 1
            print(f"\nDernier patient atteint ({total}/{total}). Fermez la fenêtre pour terminer.")

    @viewer.bind_key('Left', overwrite=True)
    def on_prev(v):
        """← — passer au patient précédent."""
        idx[0] -= 1
        if idx[0] > -1:
            charger_patient(v, idx[0])
        else:
            idx[0] += 1
            print(f"\nPremier patient atteint (1/{total}). Fermez la fenêtre pour terminer.")
    
    # Napari lie par défaut certains raccourcis sur les Labels layers
    # (ex: 'f'=fill, 'p'=paint) qui entreraient en conflit avec F/P.
    # Retrait par NOM de touche plutôt que par index positionnel dans le
    # dict — un index dépend de l'ordre d'insertion interne de napari,
    # qui peut changer d'une version à l'autre.
    couche_masque = viewer.layers[LABEL_LAYER_NAME]
    for touche in list(couche_masque.class_keymap.keys()):
        if str(touche) in ('F', 'P'):
            couche_masque.class_keymap.pop(touche, None)
    
    @viewer.bind_key('f', overwrite=True)
    def on_face(v):
        """F — marquer la vue courante comme FACE, sauvegarder."""
        pid, _ = patients[idx[0]]
        annotations[pid]['est_profil'] = False
        rafraichir_titre(v, idx[0])
        sauvegarder_json(annotations,json_path)
        print(f"  Orientation -> FACE ({pid})")
    
    @viewer.bind_key('p', overwrite=True)
    def on_profil(v):
        """P — marquer la vue courante comme PROFIL, sauvegarder."""
        pid, _ = patients[idx[0]]
        annotations[pid]['est_profil'] = True
        rafraichir_titre(v, idx[0])
        sauvegarder_json(annotations,json_path)
        print(f"  Orientation -> PROFIL ({pid})")
    
    @viewer.bind_key('q', overwrite=True)
    def on_quit(v):
        """Q — sauvegarder et fermer la session."""
        from qtpy.QtWidgets import QMessageBox
    
        reponse = QMessageBox.question(
            None, "Quitter", "Sauvegarder avant de quitter ?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        if reponse == QMessageBox.StandardButton.Cancel:
            print("Fermeture annulée.")
            return
        if reponse == QMessageBox.StandardButton.Yes:
            sauvegarder_courant(v)
            print("Session sauvegardée.")
        else:
            print("Session non sauvegardée.")
        print(f"\nSession terminée à [{idx[0]+1}/{total}].")
        v.close()
    
    # ── Un seul appel à napari.run() ──────────────────────────────────────
    napari.run()
    print(f"\nSession terminée. {idx[0]+1}/{total} patients traités.")
    return annotations

# ─────────────────────────────────────────────────────────────
# 3. EXPORT VERS nnU-Net
# ─────────────────────────────────────────────────────────────

def exporter_nnunet(
    annotations_json: str,
    output_dir: str,
    dataset_id: int = 1,
    dataset_name: str = "Protheses",
    test_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    """
    Convertit les annotations corrigées au format nnU-Net v2.

    Structure produite :
        nnUNet_raw/
        └── Dataset001_Protheses/
            ├── imagesTr/      ← images train  (caseXXXX_0000.nii.gz)
            ├── labelsTr/      ← masques train  (caseXXXX.nii.gz)
            ├── imagesTs/      ← images test    (caseXXXX_0000.nii.gz)
            ├── labelsTs/      ← masques test  (caseXXXX.nii.gz)
            └── dataset.json   ← métadonnées nnU-Net
            └── gt.json   ← json de la verité terrain

    Args:
        annotations_json: Chemin vers annotations.json.
        output_dir:       Dossier nnUNet_raw de destination.
        dataset_id:       Identifiant numérique du dataset (ex: 1 → Dataset001).
        dataset_name:     Nom du dataset (ex: "Protheses").
        test_ratio:       Fraction réservée au test.
        seed:             Graine pour le split reproductible.
    """
    annotations = charger_json(annotations_json)
    gt = {}
 
    valides = {
        pid: data for pid, data in annotations.items()
        if data.get('statut') == 'validé'
        and Path(data.get('mask_corrected', '')).exists()
    }
 
    if not valides:
        print("Aucun cas validé trouvé.")
        return
 
    print(f"{len(valides)} cas validés disponibles.")
 
    # --- Précalcul des strates (prothese, profil) AVANT le split ---
    
    strata: dict[str, tuple[bool, bool]] = {}
 
    for pid, data in valides.items():
        a_prothese = bool(data.get('prothese_auto', False))
        est_profil = bool(data['est_profil'])
        strata[pid] = (a_prothese, est_profil)
 
    # --- Bilan des strates avant split, pour vérification/log ---
    counts = defaultdict(int)
    for key in strata.values():
        counts[key] += 1
    print("Répartition par strate (prothese, profil) avant split :")
    for key, n in sorted(counts.items()):
        print(f"  prothese={key[0]!s:<5} profil={key[1]!s:<5} -> {n} cas")
 
    # --- Split train/test stratifié  ---
    train_ids, test_ids = _stratified_split(strata, test_ratio, seed)
 
    # Création des dossiers nnU-Net
    dataset_folder = f"Dataset{dataset_id:03d}_{dataset_name}"
    out = Path(output_dir) / dataset_folder
    for sub in ('imagesTr', 'labelsTr', 'imagesTs', 'labelsTs'):
        (out / sub).mkdir(parents=True, exist_ok=True)
 
    gt_json_path = out / "gt.json"
 
    # Copie avec nomenclature nnU-Net 
    for split, ids in [('train', train_ids), ('test', test_ids)]:
        for idx, pid in enumerate(sorted(ids)):
            data = valides[pid]
            case_id = f"case_{idx:04d}"
            gt_id = case_id
            img_path = str(out / 'imagesTr' / f"{case_id}_0000.nii.gz") if split == 'train' else str(out / 'imagesTs' / f"{case_id}_0000.nii.gz")
            msk_path = str(out / 'labelsTr' / f"{case_id}.nii.gz") if split == 'train' else str(out / 'labelsTs' / f"{case_id}.nii.gz")
 
 
            img = sitk.ReadImage(data['image'])
            size = list(img.GetSize())
            size[2] = 0
            sitk.WriteImage(sitk.Extract(img, size, [0, 0, 0]), img_path)
 
            msk = sitk.ReadImage(data['mask_corrected'])

            if split == 'test':
                gt[gt_id] = {
                    'image': img_path,
                    'mask': msk_path,
                    'prothese': data['prothese_auto'],
                    'lateralite': data['lateralite_auto'],
                    'bboxes_list': data['bboxes_list'],
                }
                est_profil = data['est_profil']
    
                gt[gt_id]['lateralite'] = calculer_lateralite(
                    extraire_bboxes(msk, SURFACE_MIN_MM2), img.GetSize()[0], est_profil
                )
                gt[gt_id]['prothese'] = gt[gt_id]['lateralite'] != 0
                gt[gt_id]['est_profil'] = est_profil  # traçabilité de la strate utilisée
 
            size = list(msk.GetSize())
            size[2] = 0
            sitk.WriteImage(sitk.Extract(msk, size, [0, 0, 0]), msk_path)
 
    # dataset.json obligatoire pour nnU-Net v2
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "prothese": 1},
        "numTraining": len(train_ids),
        "file_ending": ".nii.gz",
        "dataset_name": dataset_folder,
    }
    sauvegarder_json(dataset_json, out / 'dataset.json')
    sauvegarder_json(gt, gt_json_path)
 
    # --- Bilan final par strate (pour vérifier que le split a bien tenu) ---
    print(f"  Train : {len(train_ids)} cas -> {out / 'imagesTr'}")
    print(f"  Test  : {len(test_ids)}  cas -> {out / 'imagesTs'}")
    print("Vérification du ratio test par strate :")
    for key in sorted(counts.keys()):
        n_train = sum(1 for pid in train_ids if strata[pid] == key)
        n_test = sum(1 for pid in test_ids if strata[pid] == key)
        total = n_train + n_test
        ratio = n_test / total if total else float("nan")
        print(f"  prothese={key[0]!s:<5} profil={key[1]!s:<5} "
              f"-> train={n_train:>4} test={n_test:>4} ratio_test={ratio:.3f}")
 
    print("  Prêt pour :")
    print(f"     nnUNetv2_plan_and_preprocess -d {dataset_id:03d} --verify_dataset_integrity")


# ─────────────────────────────────────────────────────────────
# FONCTIONS PRIVÉES UTILITAIRES
# ─────────────────────────────────────────────────────────────

def _extraire_patient_id(nifti_path: str) -> str:
    """Extrait l'identifiant patient depuis un chemin NIfTI."""
    return Path(nifti_path).name.removesuffix('.nii.gz').removesuffix('.nii')


def _sitk_to_array(source: str | sitk.Image | None) -> np.ndarray | None:
    """Convertit un chemin ou une image SimpleITK en array numpy."""
    if source is None:
        return None
    if isinstance(source, str):
        return sitk.GetArrayFromImage(sitk.ReadImage(source))
    return sitk.GetArrayFromImage(source)


def _sauvegarder_masque_corrige(
    viewer,
    patient_id: str,
    info: dict,
    img_ref: sitk.Image,
    annotations: dict,
    json_path: Path,
) -> bool:
    """
    Extrait le masque corrigé du viewer Napari, le sauvegarde en NIfTI
    et met à jour annotations.json.
    """
    if LABEL_LAYER_NAME not in viewer.layers:
        logger.warning(f"Layer '{LABEL_LAYER_NAME}' absent pour {patient_id}")
        return False
    

    data = viewer.layers[LABEL_LAYER_NAME].data.astype(np.uint8)

    img_corr = sitk.GetImageFromArray(data)
    img_corr.CopyInformation(img_ref)
    sitk.WriteImage(img_corr, info['mask_corrected'])
    
    bboxes_list = extraire_bboxes(img_corr)
    longueur_x = img_corr.GetSize()[0]
    est_profil = annotations[patient_id]['est_profil']
    if est_profil == None:
        logger.warning(f"Latéralité indéterminé pour {patient_id}")
        return False
    lateralite = calculer_lateralite(bboxes_list, longueur_x, est_profil)

            
    prothese_real = lateralite != 0
    
    
    info['bboxes_list'] = bboxes_list
    info['statut'] = 'validé'
    info['lateralite_auto'] = lateralite
    info['prothese_auto'] = prothese_real
    annotations[patient_id] = info
    sauvegarder_json(annotations, json_path)
    
    return True


def _stratified_split(
    strata: dict[str, tuple[bool, bool]],
    test_ratio: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    """
    Split train/test qui respecte test_ratio À L'INTÉRIEUR de chaque
    combinaison (prothese, profil), pas seulement globalement.
 
    strata : {pid: (a_prothese: bool, est_profil: bool | None)}
 
    Pour une strate à un seul élément, ce cas est placé en train par
    défaut (on ne peut pas prélever un test représentatif sur 1 seul cas).
    """
    rng = np.random.default_rng(seed)
    groups: dict[tuple[bool, bool], list[str]] = defaultdict(list)
    for pid, key in strata.items():
        groups[key].append(pid)
 
    train_ids: set[str] = set()
    test_ids: set[str] = set()
    for key, ids in groups.items():
        ids = sorted(ids)  # ordre déterministe avant shuffle -> reproductible
        rng.shuffle(ids)
        n_test = max(1, round(len(ids) * test_ratio)) if len(ids) > 1 else 0
        test_ids.update(ids[:n_test])
        train_ids.update(ids[n_test:])
 
    return train_ids, test_ids


def _afficher_resume(annotations: dict, json_path: Path) -> None:
    """Affiche un résumé de la session de génération."""
    positifs = sum(1 for v in annotations.values() if v.get('prothese_auto') or v.get('prothese'))
    erreurs  = sum(1 for v in annotations.values() if v.get('statut') == 'erreur')
    print(f"\n{'─'*45}")
    print(f"  Total    : {len(annotations)}")
    print(f"  Positifs : {positifs}")
    print(f"  Négatifs : {len(annotations) - positifs - erreurs}")
    print(f"  Erreurs  : {erreurs}")
    print(f"  Suivi    : {json_path}")
    print(f"{'─'*45}")