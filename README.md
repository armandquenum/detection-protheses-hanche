# Détection de prothèses de hanche dans l’imagerie TEP chez les patients atteints d’un cancer

Détection automatique de prothèses de hanche sur topogrammes CT
(SimpleITK + nnU-Net), avec reconstruction 3D par projections multi-angles
et comparaison de méthodes (phase 2).

## Contexte

Ce projet a été développé dans le cadre d'un stage de M1 au Centre Eugène Marquis. Il vise 
à détecter et segmenter automatiquement les prothèses de hanche sur une base non filtrée 
de 34 787 topogrammes CT 2D (2021-2025, toutes régions anatomiques confondues, sans 
annotation disponible), en combinant une baseline SimpleITK et des modèles nnU-Net (2D). 
Pour une future segmentation 3D des CT patients, quatre méthodes de reconstruction d'un 
masque 3D par projections multi-angles ont été implémentées et comparées sur le dataset 
TotalSegmentator (phase 2).

Pour l'architecture technique détaillée du projet (pipeline, formats de
données, points ouverts pour la suite), voir [`ARCHITECTURE.md`](./ARCHITECTURE.md).

## Installation

Le projet est packagé comme une bibliothèque Python (`pyproject.toml`,
setuptools). Depuis la racine du dépôt :

```bash
pip install -e .
```

L'option `-e` (editable) installe le paquet en mode développement : les
modifications faites dans `src/` sont prises en compte sans réinstallation.
Pour une installation figée (ex. déploiement), utiliser `pip install .`
à la place.

Python **3.10 ou supérieur** est requis. Les dépendances Python
(`SimpleITK`, `numpy`, `scipy`, `matplotlib`, `tqdm`, `pyyaml`) sont
installées automatiquement à partir de `pyproject.toml`.

Dépendances externes non-Python (à installer séparément) : `plastimatch`
(conversion DICOM → NIfTI), `nnU-Net` (v1 et v2, voir `MODELS/nnunet/` et
`MODELS/nnunetv2/`).

## Utilisation rapide
## Utilisation rapide

### 1. Prétraitement — DICOM → NIfTI

Convertit un dossier de fichiers DICOM en NIfTI via Plastimatch.

```bash
python scripts/run_preprocessing.py [options]
```
<table>
<tr><th style="width:200px">Option</th><th style="width:80px">Alias</th>
  <th style="width:80px">Type</th><th style="width:80px">Défaut</th>
  <th style="width:80px">Description</th></tr>
<tr><td>`--dicoms_parent_folder`</td><td>`-d`</td><td>str</td><td>`DATA_RAW` (config.yaml)</td><td>Dossier racine contenant les DICOM à convertir</td></tr>
</table>

| Option&nbsp; | Alias | Type | Défaut | Description |
|---|---|---|---|---|
| `--&nbsp;dicoms_parent_folder` | `-d` | str | `DATA_RAW` (config.yaml) | Dossier racine contenant les DICOM à convertir |
| `--niftis_parent_folder` | `-n` | str | `DATA_NIFTIS` (config.yaml) | Dossier racine de destination des NIfTI générés |
| `--old_dicom_paths` | `-e` | str | `None` | Chemin vers un `.npy` de chemins DICOM déjà convertis, à exclure (reprise après interruption) |
| `--n_workers` | `-w` | int | `4` | Nombre de threads parallèles pour la conversion (4-8 selon le CPU) |

Exemple — reprise après interruption, 8 threads :
\```bash
python scripts/run_preprocessing.py -d data/raw -n data/processed/niftis -e data/dicom_paths.npy -w 8
\```

### 2. Inférence

Lance un modèle sur des images NIfTI et sort un résumé CSV avec une ligne
par patient (présence de prothèse, latéralité, métadonnées DICOM).

\```bash
python scripts/run_inference.py [options]
\```

| Option | Alias | Type | Défaut | Description |
|---|---|---|---|---|
| `--input` | `-i` | str (liste) | `DATA_NIFTIS` (config.yaml) | Dossier racine contenant les images, ou liste de fichiers `.nii.gz` |
| `--output` | `-o` | str | `DATA_NNUNET_MASK` (config.yaml) | Dossier de destination des masques prédits et du CSV de résumé |
| `--model` | `-m` | str | `nnunetv2` | Modèle à utiliser : `nnunet`, `nnunetv2` ou `sitk` |
| `--dataset_id` | — | int | `1` | ID numérique du dataset nnU-Net, ignoré pour `sitk` |
| `--fold` | `-f` | int (liste) | `0 1 2 3 4` | Folds à utiliser pour les modèles nnU-Net, ignoré pour `sitk` |
| `--device` | `-d` | str | `cpu` | Device utilisé pour l'inférence : `cpu` ou `cuda` |

Exemple — liste explicite d'images, nnU-Net v1, sur GPU :
\```bash
python scripts/run_inference.py -i data/niftis/case1.nii.gz data/niftis/case2.nii.gz -m nnunet -d cuda
\```

### 3. Évaluation

Évalue un modèle (nnU-Net v1/v2 ou baseline SimpleITK) sur un dataset de
test au format nnU-Net et sort un résumé CSV avec les métriques par
patient.

\```bash
python scripts/run_evaluation.py [options]
\```

| Option | Alias | Type | Défaut | Description |
|---|---|---|---|---|
| `--input` | `-i` | str | dataset nnU-Netv2 par défaut | Dossier racine contenant le dataset de test au format nnU-Net |
| `--dataset_id` | — | int | `1` | ID numérique du dataset nnU-Net (ex : `Dataset001_xxx` → `1`), utilisé pour `nnunet`/`nnunetv2` |
| `--model` | `-m` | str | `nnunetv2` | Modèle à évaluer : `nnunet`, `nnunetv2` ou `sitk` |
| `--fold` | `-f` | int (liste) | `0 1 2 3 4` | Liste des folds à utiliser pour les modèles nnU-Net, ex : `-f 0 1 2` |
| `--device` | `-d` | str | `cpu` | Device utilisé pour l'inférence : `cpu` ou `cuda` |
| `--lancer_inference` / `--no-lancer_inference` | `-l` | bool | `True` | Lance l'inférence avant l'évaluation. Utiliser `--no-lancer_inference` si l'inférence a déjà été faite |

Exemple — évaluer nnU-Net v1, fold 0 uniquement, sur GPU, sans relancer l'inférence :
\```bash
python scripts/run_evaluation.py -m nnunet -f 0 -d cuda --no-lancer_inference
\```

## Résultats

### Détection / segmentation 2D — nnU-Net vs SimpleITK

Comparaison sur le jeu de test (`evaluator.evaluer_dataset_test`),
métriques dans `reports/figures/evaluation_{model}_{dataset}.csv`.

* Segmentation

| Modèle | Dice | IoU | HD95 (mm) |
|---|---|---|---|
|  SimpleITK (baseline)  | 0.848 | 0.738 | 8.195 |
| n nU-Net v1  | 0.986 | 0.975 | 3.871 |
|  nnU-Net v2  | **0.995** | **0.990** | **0.662** |

* Détection

|  Modèle  |  Sensibilité  |  Spécificité  |  Accuracy  |
|---|---|---|---|
|  SimpleITK (baseline)  | 0.441 | 0.932 | 0.695 |
|  nnU-Net v1  | 1.000 | 0.452 | 0.716 |
|  nnU-Net v2  | **1.000** | **1.000** | **1.000** |


### Reconstruction 3D — comparaison des 4 méthodes

Comparaison sur le dataset TotalSegmentator (`reconstruction_methods.comparer_methodes`).

| Méthode | Dice | IoU | HD95 (mm) |
|---|---|---|---|
| tube_plein | **0.949** | **0.904** | **0.239** |
| argmax_filtre | 0.202 | 0.156 | 9.055|
| seuil_hu | 0.840 | 0.731 | 1.717 |
| space_carving | 0.848 | 0.743 | 1.485 |

_(voir [`ARCHITECTURE.md` §2](./ARCHITECTURE.md#2-notre-approche-de-la-segmentation-3d)
pour le détail de chaque méthode)_

## État d'avancement

Détection/segmentation 2D et évaluation nnU-Net vs SimpleITK : **fait**.
Reconstruction 3D par projections multi-angles : **fait** (comparaison
des 4 méthodes). Localisation 3D (BBox), calcul SUV, corrélation
clinique : **en cours / à poursuivre**, voir
[`ARCHITECTURE.md` §5](./ARCHITECTURE.md#5-points-ouverts--todo-pour-la-suite)
pour le détail de ce qui reste à faire et par où continuer.
