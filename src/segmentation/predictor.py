# src/segmentation/predictor.py

import os
import subprocess
import logging
import shutil
import re
import SimpleITK as sitk
import numpy as np
from pathlib import Path


from src.utils.io import (
    charger_dicom_paths,
    charger_json, sauvegarder_json,
    lire_metadata_dicom,
    calculer_lateralite,
    extraire_bboxes,
    detecter_vue_profil_dicom,
    resoudre_dicom_path,
    dossier_predictions,
)
from src.utils.paths import ROOT, MODELS, DATA_NNUNET_MASK

logger = logging.getLogger(__name__)

# Chemin relatif au dossier du modèle actif (voir configurer_env_nnunet),
# utilisé uniquement par l'étape de postprocessing nnU-Net v1.
# NOTE : NNUNET/NNUNETV2 gérés séparément par l'utilisateur dans paths.py.
POSTPROCESSING = "nnUNet_results/Dataset001_Protheses/nnUNetTrainer__nnUNetResEncUNetPlans__2d/crossval_results_folds_0_1_2_3_4/"


def configurer_env_nnunet(model: str = "nnunet") -> None:
    """
    Configure les variables d'environnement nnU-Net
    pour pointer vers les dossiers du projet.
    Doit être appelé avant toute commande nnU-Net.

    Args:
        model: Nom du modèle ('nnunet' ou 'nnunetv2'), utilisé pour
               construire les chemins nnUNet_raw/nnUNet_preprocessed/
               nnUNet_results propres à ce modèle.
    """
    os.environ["nnUNet_raw"]          = str(ROOT / f"data/annotations/{model.upper()}/nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = str(ROOT / f"data/annotations/{model.upper()}/nnUNet_preprocessed")
    os.environ["nnUNet_results"]      = str(MODELS / model.lower())
    logger.debug("Variables nnU-Net configurées")


def predire_dossier_nnunet_format(
    input_dir: str,
    output_dir: str,
    dataset_id: int | str = 1,
    config: str = "2d",
    fold: list[str] | list[int] | int | str = 0,
    save_probabilities: bool = False,
    device: str = "cpu",
    postprocessing: bool = False,
) -> dict:
    """
    Lance nnUNetv2_predict sur un dossier d'images.

    Les images dans input_dir doivent respecter la nomenclature nnU-Net :
        caseXXXX_0000.nii.gz

    Args:
        input_dir:          Dossier contenant les images à prédire.
        output_dir:         Dossier de sortie des masques prédits.
        dataset_id:         ID du dataset (ex: 1 → Dataset001).
        config:             Configuration nnU-Net ('2d', '3d_fullres'...).
        fold:               Fold du modèle à utiliser (0-4 ou 'all').
        save_probabilities: Sauvegarder les cartes de probabilités (.npz).
        device:             'cpu', 'cuda' ou 'mps'.
        postprocessing:     True ou False.

    Returns:
        dict avec 'success', 'n_images', 'output_dir'.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        "nnUNetv2_predict",
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-d", str(dataset_id),
        "-c", config,
        "-f", *np.array(fold).astype(str).tolist(),
        "-tr", "nnUNetTrainer",
        "-device", device,
        "-p", "nnUNetResEncUNetPlans",
        "-chk", "checkpoint_best.pth",
    ]

    if save_probabilities:
        cmd.append("--save_probabilities")

    n_images = len(list(Path(input_dir).glob("*.nii.gz")))
    logger.info(f"Prédiction sur {n_images} images...")

    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Prédictions sauvegardées dans {output_dir}")
        if not postprocessing:
            return {"success": True, "n_images": n_images, "output_dir": output_dir}
        else:
            cmd = [
                "nnUNetv2_apply_postprocessing",
                "-i", str(output_dir),
                "-o", str(output_dir),
                "-pp_pkl_file", str(POSTPROCESSING + "postprocessing.pkl"),
                "-np", "8",
                "-plans_json", str(POSTPROCESSING + "plans.json"),
            ]

            try:
                subprocess.run(cmd, check=True)
                logger.info(f"Prédictions + Postprocessing sauvegardées dans {output_dir}")
                return {"success": True, "n_images": n_images, "output_dir": output_dir}
            except subprocess.CalledProcessError as e:
                logger.error(f"Erreur nnUNetv2_apply_postprocessing : {e}")
                return {"success": False, "n_images": n_images, "output_dir": output_dir}
    except subprocess.CalledProcessError as e:
        logger.error(f"Erreur nnUNetv2_predict : {e}")
        return {"success": False, "n_images": n_images, "output_dir": output_dir}


def predire_image_unique(
    image: sitk.Image | str,
    dataset_id: int | str = 1,
    config: str = "2d",
    fold: int = 0,
    device: str = "cpu",
    tmp_dir: str = "/tmp/nnunet_predict",
) -> sitk.Image:
    """
    Lance une prédiction sur une seule image SimpleITK.
    Crée un dossier temporaire, lance nnU-Net, retourne le masque.

    Args:
        image:      Image SimpleITK ou chemin NIfTI.
        dataset_id: ID du dataset.
        config:     Configuration nnU-Net.
        fold:       Fold du modèle.
        device:     Dispositif de calcul.
        tmp_dir:    Dossier temporaire (nettoyé après).

    Returns:
        Masque binaire prédit (sitk.Image).
    """
    tmp = Path(tmp_dir)
    tmp_in  = tmp / "input"
    tmp_out = tmp / "output"
    tmp_in.mkdir(parents=True, exist_ok=True)
    tmp_out.mkdir(parents=True, exist_ok=True)

    try:
        # Sauvegarder l'image au format nnU-Net
        if isinstance(image, str):
            image = sitk.ReadImage(image)
        size = list(image.GetSize())
        size[2] = 0
        tmp_img = tmp_in / "case_0000_0000.nii.gz"
        sitk.WriteImage(sitk.Extract(image, size, [0, 0, 0]), str(tmp_img))

        # Prédire
        result = predire_dossier_nnunet_format(
            str(tmp_in), str(tmp_out),
            dataset_id=dataset_id, config=config,
            fold=fold, device=device
        )

        if not result["success"]:
            raise RuntimeError("Prédiction nnU-Net échouée")

        # Lire le masque prédit
        masque_path = tmp_out / "case_0000.nii.gz"
        masque = sitk.ReadImage(str(masque_path))
        masque = sitk.JoinSeries([masque])
        masque.CopyInformation(image)
        return masque

    finally:
        # Nettoyage du dossier temporaire
        shutil.rmtree(tmp, ignore_errors=True)


def predire_images_multiple(
    images: list[str] | None = None,
    masks_folder: str = DATA_NNUNET_MASK,
    dataset_id: int | str = 1,
    config: str = "2d",
    fold: list[str] | list[int] | str | int = [0, 1, 2, 3, 4],
    device: str = "cpu",
    tmp_dir: str = "/tmp/nnunet_predict",
    dataset_path: str = "",
    mode: str = "infer",
    version: int = 1,
) -> dict:
    """
    Lance une prédiction sur une liste d'images NIfTI.
    Crée un dossier temporaire, lance nnU-Net, retourne le masque.

    Le paramètre mode sélectionne le comportement :
            - Inférence:  mode='infer', lance une inférence sur `images`.
            - Évaluation : mode='eval', évalue nnU-Net sur un dataset déjà
              constitué (voir exporter_nnunet) — nécessite `dataset_path`
              pointant vers un dossier DatasetXXX_Nom contenant gt.json.

    Args:
        images:       Liste de chemins NIfTI (mode='infer' uniquement).
        masks_folder: Dossier dans lequel sont enregistrés les masques (mode='infer').
        dataset_id:   ID du dataset (mode='infer').
        config:       Configuration nnU-Net.
        fold:         Fold du modèle.
        device:       Dispositif de calcul.
        tmp_dir:      Dossier temporaire (nettoyé après), mode='infer'.
        dataset_path: Chemin du dataset nnU-Net DatasetXXX_Nom (mode='eval').
        mode:         'infer' ou 'eval', voir description ci-dessus.
        version:      1 ou 2, sélectionne le sous-dossier de sortie
                      ('nnunet_seg' ou 'nnunetv2_seg') en mode='eval'.

    Returns:
        Dictionnaire de segmentations par patient/case_id, ou {} si le
        mode ou le dataset_path (mode='eval') sont invalides.
    """
    images = images or []

    if mode not in ('infer', 'eval'):
        logger.warning(f"mode: {mode} non reconnu, choisir entre ('infer'/'eval')")
        return {}

    if mode == 'infer':
        tmp = Path(tmp_dir)
        masques_folder = Path(masks_folder)

        if not masques_folder.exists():
            masques_folder.mkdir(parents=True, exist_ok=True)

        segmentations = charger_json(masques_folder / 'segmentations.json')

        deja_traites = [v['image'] for pid, v in segmentations.items()]
        print(f"Images déja traités : {len(deja_traites)}")

        images_path = [img for img in images if not img in deja_traites]

        tmp_in, tmp_out = _export_to_nnunet_input_format(images_path, tmp)
        try:
            # Prédire
            result = predire_dossier_nnunet_format(
                str(tmp_in), str(tmp_out),
                dataset_id=dataset_id, config=config,
                fold=fold, device=device
            )
            result["output_dir"] = masques_folder
            print(result)

        except Exception as e:
            # Une erreur inattendue dans predire_dossier_nnunet_format ne doit
            # pas empêcher le nettoyage (finally) ni faire planter la fonction
            # sans retour exploitable — on logue et on continue vers le
            # post-traitement (qui gérera lui-même l'absence de résultats).
            logger.error(f"Erreur lors de la prédiction nnU-Net (mode infer) : {e}")

        finally:
            # lire et enregistrer les masques sous le bon format
            correspondance = charger_json(tmp / "correspondance.json")
            dicom_paths = charger_dicom_paths()
            for _, info in correspondance.items():
                patient_id = Path(info['image_orginale']).name.removesuffix('.nii.gz')
                try:
                    masque_path = masques_folder / f"{patient_id}_mask.nii.gz"
                    masque_predite_path = Path(info['predicted_mask'])
                    if not masque_predite_path.exists():
                        logger.error(f"Masque prédit introuvable pour {patient_id} : {masque_predite_path}")
                        continue

                    masque = sitk.ReadImage(str(masque_predite_path))
                    img_original = sitk.ReadImage(info['image_orginale'])
                    if len(img_original.GetSize()) == 3:
                        masque = sitk.JoinSeries([masque])

                    masque.CopyInformation(img_original)
                    sitk.WriteImage(masque, masque_path)

                    dicom_path = resoudre_dicom_path(patient_id, dicom_paths)
                    est_profil = detecter_vue_profil_dicom(dicom_path)['est_profil']

                    bboxes_list = extraire_bboxes(masque)
                    longueur_x = masque.GetSize()[0]
                    lateralite = calculer_lateralite(bboxes_list, longueur_x, est_profil)

                    prothese = lateralite != 0

                    segmentations[patient_id] = {
                        'image':       str(info['image_orginale']),
                        'mask':        str(masque_path),
                        'prothese':    prothese,
                        'lateralite':  lateralite,
                        'bboxes_list': bboxes_list,
                        'est_profil':  est_profil,
                    }

                    if dicom_path:
                        segmentations[patient_id].update(lire_metadata_dicom(dicom_path))

                except Exception as e:
                    logger.error(f"Erreur sur {patient_id} : {e}")
                    segmentations[patient_id] = {'statut': 'erreur', 'message': str(e)}

            sauvegarder_json(segmentations, masques_folder / "segmentations.json")
            # Nettoyage du dossier temporaire
            shutil.rmtree(tmp, ignore_errors=True)
        return segmentations

    else:
        path_of_dataset = Path(dataset_path)
        dataset_name = path_of_dataset.name
        is_dataset_name = re.search(r"^Dataset\d{3}_[a-zA-Z0-9_]+$", dataset_name)
        if path_of_dataset.exists() and is_dataset_name:
            gt_json = charger_json(path_of_dataset / 'gt.json')
            images_folder = path_of_dataset / 'imagesTs'
            masques_folder = dossier_predictions('nnunet', path_of_dataset, version=version)
            if not masques_folder.exists():
                masques_folder.mkdir(parents=True, exist_ok=True)

            segmentations = charger_json(masques_folder / 'segmentations.json')

            deja_traites = [v['image'] for _, v in segmentations.items()]
            print(f"Images déja traités : {len(deja_traites)}")

            deja_traites_folder = path_of_dataset / 'deja_traites'

            if not deja_traites_folder.exists() and len(deja_traites) > 0:
                deja_traites_folder.mkdir(parents=True, exist_ok=True)

            new_deja_traites_paths = [
                shutil.move(
                    img,
                    img.replace(str(images_folder), str(deja_traites_folder))
                ) for img in deja_traites
            ]

            if len(new_deja_traites_paths) != len(deja_traites):
                raise RuntimeError(
                    f"Déplacement incomplet des cas déjà traités : "
                    f"{len(new_deja_traites_paths)}/{len(deja_traites)} déplacés"
                )

            try:
                result = predire_dossier_nnunet_format(
                    str(images_folder), str(masques_folder),
                    dataset_id=dataset_name, config=config,
                    fold=fold, device=device
                )
                print(result)

            except Exception as e:
                # Même logique que pour le mode infer : ne pas laisser une
                # erreur inattendue empêcher la restauration des cas déjà
                # traités ni le nettoyage (finally).
                logger.error(f"Erreur lors de la prédiction nnU-Net (mode eval) : {e}")

            finally:
                for i in range(len(deja_traites)):
                    shutil.move(new_deja_traites_paths[i], deja_traites[i])
                shutil.rmtree(deja_traites_folder, ignore_errors=True)
                [os.remove(masques_folder / f"{file}.json") for file in ["dataset", "predict_from_raw_data_args", "plans"]]


                for case_id, info in gt_json.items():
                    try:
                        image_originale_path = info['image']

                        masque_path = masques_folder / f"{case_id}.nii.gz"
                        if not masque_path.exists():
                            logger.error(f"Masque prédit introuvable pour {case_id} : {masque_path}")
                            continue

                        masque = sitk.ReadImage(str(masque_path))
                        img_original = sitk.ReadImage(image_originale_path)
                        if len(img_original.GetSize()) == 3:
                            masque = sitk.JoinSeries([masque])

                        masque.CopyInformation(img_original)
                        sitk.WriteImage(masque, masque_path)

                        est_profil = info['est_profil']

                        bboxes_list = extraire_bboxes(masque,1)
                        longueur_x = masque.GetSize()[0]
                        lateralite = calculer_lateralite(bboxes_list, longueur_x, est_profil)

                        prothese = len(bboxes_list) != 0

                        segmentations[case_id] = {
                            'image':       str(image_originale_path),
                            'mask':        str(masque_path),
                            'prothese':    prothese,
                            'lateralite':  lateralite,
                            'bboxes_list': bboxes_list,
                            'est_profil':  est_profil,
                        }
                    except Exception as e:
                        logger.error(f"Erreur sur {case_id} : {e}")
                        segmentations[case_id] = {'statut': 'erreur', 'message': str(e)}

                sauvegarder_json(segmentations, masques_folder / "segmentations.json")
            return segmentations

        else:
            logger.warning(f"{dataset_path} : dossier introuvable ou nom invalide (attendu 'DatasetXXX_Nom')")
            return {}


def _export_to_nnunet_input_format(
        nifti_paths: list[str],
        tmp_dir: str | Path) -> tuple[str | Path, str | Path]:
    """
    Crée un dossier temporaire qui respecte le format d'entrée de nnU-Net
    pour la prédiction.

    Structure de sortie :
    tmp_dir/
    ├── input/                ← input pour nnU-Net
    ├── output/               ← output de nnU-Net
    └── correspondance.json   ← correspondance entre l'image format nnU-Net et l'image originale

    Args:
        nifti_paths: Liste des chemins NIfTI patients.
        tmp_dir:     Dossier de travail pour l'annotation.

    Returns:
        Tuple (tmp_in, tmp_out) : chemins des dossiers d'entrée/sortie
        nnU-Net créés dans tmp_dir.
    """
    tmp = Path(tmp_dir)
    tmp_in  = tmp / "input"
    tmp_out = tmp / "output"
    tmp_in.mkdir(parents=True, exist_ok=True)
    tmp_out.mkdir(parents=True, exist_ok=True)

    correspondance = {}

    for idx, path in enumerate(nifti_paths):
        case_id = f"case_{idx:04d}"

        img = sitk.ReadImage(path)
        size = list(img.GetSize())
        if len(size) == 3:
            size[2] = 0

        sitk.WriteImage(
            sitk.Extract(img, size, [0] * len(size)),
            str(tmp_in / f"{case_id}_0000.nii.gz")
        )

        correspondance[case_id] = {
            'image_orginale': str(path),
            'predicted_mask': str((tmp_out / f"{case_id}.nii.gz").absolute()),
        }

    sauvegarder_json(correspondance, tmp / "correspondance.json")

    return (tmp_in, tmp_out)