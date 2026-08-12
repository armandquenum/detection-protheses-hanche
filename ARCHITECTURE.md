# Architecture du projet — Détection de prothèses de hanche

Document de passation. Objectif : permettre à la personne qui reprend le
projet de comprendre le pipeline, les formats de données, les pièges déjà
corrigés, et ce qu'il reste à construire, sans avoir à relire tout
l'historique de développement.

---

## 1. Vue d'ensemble du pipeline

```
DICOM (données brutes)
    │
    ▼
[1] loader.py : run_conversion_pipeline()
    → conversion DICOM → NIfTI (Plastimatch)
    → dicom_paths.npy (index des chemins DICOM, sous DATA/)
    │
    ▼
[2] pipeline_annotation.py : generer_annotations_candidates(mode='annotation', merge_into_annotations=True)
    → masques candidats (SimpleITK) + détection orientation (face/profil)
    → annotations.json  (à corriger manuellement via Napari)
    │
    ▼
[3] pipeline_annotation.py : exporter_nnunet()
    → dataset nnU-Net (imagesTr/labelsTr/imagesTs/labelsTs)
    → gt.json  (vérité terrain, cas de TEST uniquement)
    │
    ▼
[4] Entraînement nnU-Net (train.py, hors périmètre de cette doc)
    │
    ▼
[5] Inférence / évaluation — 4 chemins possibles, même format de sortie :
    ├── predictor.py : predire_images_multiple(mode='infer')
    │     → tous les patients, hors évaluation
    ├── predictor.py : predire_images_multiple(mode='eval')
    │     → nnU-Net (v1/v2) sur le jeu de test
    ├── pipeline_annotation.py : generer_annotations_candidates(mode='eval')
    │     → baseline SimpleITK sur le jeu de test
    └── pipeline_annotation.py : generer_annotations_candidates(mode='annotation', merge_into_annotations=False)
          → baseline SimpleITK ad-hoc (ex. scripts/run_inference.py -m sitk),
            sur des images arbitraires hors jeu de test
    Les 4 écrivent segmentations.json (même schéma de champs) dans
    output_dir/dossier_predictions selon le chemin
    │
    ▼
[6] evaluator.py : evaluer_dataset_test()
    → compare segmentations.json à gt.json
    → metrics.toutes_les_metriques() (Dice, IoU, HD95, latéralité...)
    → CSV (reports/figures/evaluation_{model}_{dataset}.csv) + résumé console
```

### Phase 2 (partiellement commencée)

```
localisation3d/bounding_box3d.py
    → BBox 2D coronale (face) + BBox 2D sagittale (profil, pré-découpée)
    → BBox 3D par côté (gauche/droite)
    │
    ▼  [MAILLON MANQUANT — voir §5.1]
    │
reconstruction3d/roi_projection.py + reconstruction_methods.py
    → projection multi-angles (roi_projection.projeter) sur ROI 3D —
      roi_projection.py couvre uniquement 3D->2D et sa préparation
    → rétroprojection, reconstruction 3D et 4 méthodes comparées
      (tube_plein, argmax_filtre, seuil_hu, space_carving) sur le
      dataset TotalSegmentator — tout ce qui va de 2D->3D vit dans
      reconstruction_methods.py (reconstruct_3d, comparer_methodes)
    → métriques (dice_score/iou_score/hausdorff_distance, centralisées
      dans src.evaluation.metrics)
```

Tout ce qui suit (TotalSegmentator sur les vraies données patient, calcul
SUV, corrélation clinique) n'est pas commencé — voir §5.

---

## 2. Notre approche de la segmentation 3D

Les modèles nnU-Net du projet sont entraînés sur des **topogrammes 2D**
(images de type radiographie, face/profil), pas sur des volumes CT 3D
complets. Pour obtenir malgré tout un masque 3D de la prothèse, l'approche
retenue est une **reconstruction par projections multi-angles**, plutôt
qu'un modèle de segmentation 3D natif. La projection (3D→2D) et la
reconstruction (2D→3D) sont dans deux fichiers séparés :
`roi_projection.py` (projection uniquement) et
`reconstruction_methods.py` (tout le reste).

1. **Projection** (`roi_projection.projeter`) : depuis un volume CT 3D
   (ROI extraite), on génère des projections 2D à plusieurs angles
   (ex. 0°, 30°, 60°... — équivalent numérique d'un topogramme pris sous
   différents angles).
2. **Segmentation 2D** : chaque projection est segmentée par le modèle
   nnU-Net (2D), comme un topogramme normal.
3. **Rétroprojection** (`reconstruction_methods.retroprojeter_masque`) :
   chaque masque 2D est "extrudé" en arrière dans l'espace 3D le long de
   l'angle de projection correspondant.
4. **Reconstruction 3D** (`reconstruction_methods.reconstruct_3d`,
   paramètre `method`) : les rétroprojections de tous les angles sont
   combinées pour reconstituer un volume 3D. **4 méthodes ont été
   comparées** (`comparer_methodes`, qui délègue à `reconstruct_3d` pour
   chacune) sur le dataset TotalSegmentator (qui fournit une vérité
   terrain 3D) :
   - **`tube_plein`** — extrusion uniforme sur toute la profondeur
     (référence / Visual Hull classique) : rapide, mais surestime le
     volume (garde tout ce qui est dans la silhouette 2D, sans filtrer
     par intensité). Seule des 4 méthodes purement géométrique — aucun
     besoin de l'image CT réelle.
   - **`argmax_filtre`** — ne garde, le long de chaque rayon de
     projection, que les indices dont l'intensité HU est proche du
     maximum local (fenêtre `alpha`).
   - **`seuil_hu`** — ne garde que les indices dont le HU dépasse un
     seuil physique caractéristique du métal (~1000-1500 HU), indépendant
     du max local.
   - **`space_carving`** — combine un critère géométrique (voxel présent
     dans la silhouette 2D à *tous* les angles, ou au moins `vote_ratio`
     d'entre eux) ET un critère photométrique (HU réel du voxel au-dessus
     du seuil métal) : c'est la méthode la plus proche d'une vraie
     sculpture par cohérence multi-vues, et celle qui a le moins
     tendance à surestimer le volume par rapport à `tube_plein`.
   Les 4 méthodes sont comparées par Dice, IoU et Hausdorff 95 —
   **consultez les résultats produits par `comparer_methodes` pour savoir
   laquelle a été retenue en pratique**, puis appelez
   `reconstruct_3d(..., method="<méthode retenue>")` directement pour la
   reconstruction "de production" (plus besoin de repasser par la
   comparaison une fois le choix fait). ⚠️ `reconstruct_3d` utilise
   l'image de référence passée en paramètre à la fois pour la géométrie
   et, dès que `method != "tube_plein"`, comme source d'intensités HU —
   passer la vraie image CT, pas un masque, sous peine d'échec silencieux
   (voir `evaluate_reconstruction_on_dataset` dans le même fichier pour
   l'erreur explicite correspondante).

### Rôle du second modèle nnU-Net (`nnunetv2`)

Un second modèle (`nnunetv2`, dossier `MODELS/nnunetv2/`) a été entraîné
séparément sur un **dataset plus grand et mieux équilibré** que le modèle
initial (`nnunet`). Son rôle en phase 2 : fournir des segmentations 2D
plus fiables sur les multiples angles de projection utilisés à l'étape 2
ci-dessus — ces segmentations alimentent ensuite `bounding_box3d.py` pour
construire la BBox 3D (via les vues coronale/sagittale), point de départ
de la ROI utilisée dans `roi_projection.py`. `evaluator.py` sait déjà
évaluer indifféremment `nnunet` ou `nnunetv2` (`model='nnunetv2'`) — donc
comparer les deux modèles sur les mêmes cas de test est directement
possible sans code supplémentaire.

---

---

## 3. Formats de données (JSON)

### `annotations.json` (mode='annotation', un par patient, ID brut)
```json
{
  "ACT1.cem.a22d.fr...": {
    "image": "...", "mask_auto": "...", "mask_corrected": "...",
    "prothese_auto": true, "lateralite_auto": 1, "bboxes_list": [...],
    "statut": "à_valider | validé | erreur",
    "est_profil": true | false | null,
    "jour_d_examen": "...", "date_de_naissance": "...", ...
  }
}
```

### `gt.json` (produit par `exporter_nnunet`, cas de **test uniquement**)
```json
{
  "case_0001": {
    "image": "chemin réel dans imagesTs/", "mask": "...",
    "est_profil": true|false, "lateralite": 1,
    "prothese": true,
  }
}
```

### `segmentations.json` (sortie commune du mode `eval`, tous modèles )
Ce format est également celui produit par
`generer_annotations_candidates(mode='annotation', merge_into_annotations=False)`
lors d'un appel ad-hoc (ex. `scripts/run_inference.py -m sitk`).
```json
{
  "case_0001_Ts": {
    "image": "...", "mask": "...", "prothese": true,
    "lateralite": 1, "bboxes_list": [...]
  }
}
```
Emplacement déterminé par `io.dossier_predictions(model, dataset_path, version)` :
- `sitk_seg/` — baseline SimpleITK
- `nnunet_seg/` — nnU-Net v1
- `nnunetv2_seg/` — nnU-Net v2

### Latéralité — convention entière (`io.calculer_lateralite`)
| Valeur | Sens |
|---|---|
| `-1` | orientation indéterminée (`est_profil` inconnu) |
| `0` | pas de prothèse détectée |
| `1` | gauche uniquement |
| `2` | droite uniquement |
| `3` | bilatérale |
| `4` | vue de profil (latéralité non calculable) |

**`-1` est un cas à part, jamais à confondre avec `4`** — en Python,
indexer un tableau avec `-1` boucle silencieusement sur le dernier
élément plutôt que de lever une erreur ; plusieurs endroits du code ont
dû être protégés explicitement contre ce piège (voir §3).

---

## 4. Pièges déjà corrigés (ne pas réintroduire)

Le code a été audité et plusieurs bugs silencieux corrigés :

- résolution DICOM dispersée et incohérente ;

- valeurs indéterminées `est_profil=None`/`lateralite=-1` traitées comme

  fausses/nulles sans garde-fou ;

- ordre des axes spacing/tableau numpy inversé dans les calculs de

  distance ;

- duplication de métriques et de dossiers de sortie ;

- écriture silencieuse dans `annotations.json` lors d'appels ad-hoc à

  `generer_annotations_candidates` (corrigé via le paramètre

  `merge_into_annotations` — voir §5.1 / docstring de la fonction) ;

- dans `run_inference.py` : détection de fichier via `path.is_file` sans

  parenthèses (référence de méthode toujours vraie plutôt qu'un appel,

  incluait silencieusement des fichiers inexistants ou non-NIfTI) ;

Ces corrections ont été faites dans le code actuel, mais l'historique
Git de ce dépôt ne remonte pas jusqu'à elles . **Les principes à respecter pour la suite sont
regroupés en §6** : c'est la référence à consulter pour éviter de
réintroduire ces bugs.

---

## 5. Points ouverts / TODO pour la suite

### 5.1 — Priorité haute : maillon manquant entre BBox 3D et reconstruction

`bounding_box3d.py` sait combiner une BBox 2D coronale + une BBox 2D
sagittale **déjà pré-découpée par côté** en une BBox 3D. Mais **rien
n'assemble actuellement** :
- le masque sagittal complet → deux moitiés gauche/droite pré-découpées
  (aucune fonction ne fait ce découpage aujourd'hui) ;
- `extraire_bboxes_coronale` (traite les deux côtés) + `extraire_bbox_sagittale`
  (traite un côté à la fois) → le dict `{"gauche": (bbox_coro, bbox_sagi), ...}`
  attendu par `get_3d_bboxes`.

À écrire : une fonction d'orchestration (ex. `construire_dict_2d_bboxes`)
qui fait ce pont, plus la logique de découpage du masque sagittal par
moitié (probablement par rapport au même `longueur_x`/milieu utilisé dans
`calculer_lateralite`).

### 5.2 — Lien BBox 3D → extraction de ROI

`roi_projection.py` a déjà `pad_roi_margin` pour ajouter une marge autour
d'une ROI avant projection. La BBox 3D produite par `bounding_box3d.py`
doit être utilisée pour extraire la ROI réelle (`data/rois/`, prévu dans
l'arborescence mais non implémenté) avant de lancer la projection/
reconstruction. Ce câblage est à écrire.

### 5.3 — Scripts et notebooks de la phase 2 (non commencés)
- `scripts/run_localisation_3d.py`, `notebooks/05_localisation_3d.ipynb`
- `scripts/run_totalsegmentator.py`, `notebooks/06_totalsegmentator.ipynb`,
  `src/reconstruction3d/totalsegmentator_wrapper.py`
- `scripts/generate_stage4_dataset.py`, `notebooks/07_projection_reconstruction.ipynb`
- `scripts/run_reconstruction.py`

### 5.4 — Module SUV (entièrement à écrire)
- `src/suv/dicom_suv_metadata.py` — lecture des métadonnées DICOM
  nécessaires au calcul SUV (poids patient, activité injectée, temps
  d'acquisition...). Réutiliser le pattern de `io.lire_metadata_dicom`
  (ajouter les tags SUV à la constante `METADATAS` ou créer une constante
  dédiée du même type).
- `src/suv/periprosthetic_zone.py` — dépend directement de la BBox 3D
  (§5.1/5.2) pour définir la zone péri-prothétique.
- `src/suv/suv_computation.py`
- `scripts/run_suv_pipeline.py`, `notebooks/08_suv_computation.ipynb`

### 5.5 — Corrélation clinique (entièrement à écrire)
- `src/clinical/correlation.py`
- `scripts/run_clinical_correlation.py`, `notebooks/09_correlation_clinique.ipynb`

### 5.6 — Divers, faible priorité
- `config.yaml` : section phase 2 mentionnée dans l'arborescence d'origine
  (`angles, agregation, vote_ratio, margin_mm, SUV`) pas encore ajoutée —
  actuellement ces valeurs sont passées en paramètres de fonction plutôt
  que centralisées en config.
- `__init__.py` manquants dans certains sous-dossiers de `src/`.
- `image_info.py` : docstring à compléter (mineur, fonction de debug).
- `requirements.txt` : renommage évoqué (`requierments.txt` → corrigé ou
  à vérifier).

---

## 6. Repères pour reprendre le travail

- **Avant de coder quoi que ce soit dans `suv/` ou `clinical/`**, régler
  d'abord §5.1/5.2 : c'est la dépendance commune à toute la phase 2.
- **Ne jamais réintroduire** une recherche DICOM par sous-chaîne ad hoc —
  toujours passer par `io.resoudre_dicom_path`/`resoudre_dicom_brut`.
- **Ne jamais dupliquer** une métrique (Dice/IoU/Hausdorff) — tout doit
  importer depuis `src.evaluation.metrics`.
- **Ne jamais appeler** `generer_annotations_candidates` avec
  `merge_into_annotations=True` en dehors de l'étape [2] du pipeline
  (annotation officielle avant validation Napari) — ce paramètre doit
  toujours être passé explicitement, jamais laissé au défaut, pour tout
  script ou appel exploratoire.
- **Toute nouvelle valeur "indéterminée"** (comme `est_profil=None`) doit
  être traitée comme un état explicite distinct, jamais implicitement
  convertie en `False`/`0`/`-1` sans garde-fou (`is None`, jamais `== None`
  ni `bool(x)` en amont d'un test de nullité).