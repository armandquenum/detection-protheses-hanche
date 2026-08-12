# scripts/run_evaluation.py
"""
Évalue un modèle de segmentation (nnU-Net v1/v2 ou baseline SimpleITK) sur
un dataset de test au format nnU-Net, et sort un résumé CSV avec les
métriques par patient (voir src.evaluation.evaluator.evaluer_dataset_test).

Exemples :
    # Avec les valeurs par défaut (config.yaml : NNUNETV2_DATASET, modèle
    # nnunetv2, tous les folds, CPU, avec relance de l'inférence)
    python scripts/run_evaluation.py

    # Dataset explicite, modèle nnunet, un seul fold, sur GPU
    python scripts/run_evaluation.py -i data/test_set -m nnunet -f 0 -d cuda

    # Réutiliser une inférence déjà réalisée (pas besoin de relancer
    # le modèle, on recharge juste les prédictions déjà sur le disque)
    python scripts/run_evaluation.py --no-lancer_inference
"""

import argparse

from src.evaluation.evaluator import evaluer_dataset_test
from src.utils.paths import NNUNETV2_DATASET

# Configuration nnU-Net utilisée pour l'évaluation. Un seul modèle (2D)
# est entraîné/disponible pour l'instant, donc la valeur est fixée ici
# plutôt que proposée en option en ligne de commande : exposer un choix
# qui n'en est pas un (une seule valeur possible) prête à confusion pour
# l'utilisateur. Le jour où un modèle 3D est ajouté, il suffit de
# remonter cette constante en argument --config (voir --model plus bas
# pour un exemple de argument à choix multiples).
CONFIG = "2d"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Évaluation',
        description="Évalue un modèle sur un dataset de test au format nnU-Net",
        epilog='-' * 50,
    )

    parser.add_argument(
        '-i', '--dataset_path',
        type=str,
        default=str(NNUNETV2_DATASET),
        help="Dossier racine contenant le dataset de test au format nnU-Net "
             "(défaut : NNUNETV2_DATASET dans config.yaml)."
    )
    parser.add_argument(
        '--dataset_id',
        type=int,
        default=1,
        help="ID numérique du dataset nnU-Net (ex : Dataset001_xxx -> 1). "
             "Utilisé uniquement pour les modèles 'nnunet'/'nnunetv2', "
             "ignoré pour 'sitk' (défaut : 1)."
    )
    parser.add_argument(
        '-m', '--model',
        type=str,
        default="nnunetv2",
        choices=["nnunet", "nnunetv2", "sitk"],
        help="Modèle à évaluer : 'nnunet' (v1), 'nnunetv2' ou 'sitk' "
             "(baseline SimpleITK) (défaut : nnunetv2)."
    )
    parser.add_argument(
        '-f', '--fold',
        type=int,
        nargs='*',
        default=[0, 1, 2, 3, 4],
        metavar="FOLD",
        help="Liste des folds à utiliser pour les modèles nnU-Net, "
             "ex : -f 0 1 2. Aligné sur le défaut de evaluer_dataset_test "
             "(défaut : 0 1 2 3 4, soit les 5 folds de la validation croisée)."
    )
    parser.add_argument(
        '-d', '--device',
        type=str,
        default='cpu',
        choices=["cpu", "cuda"],
        help="Device utilisé pour l'inférence : 'cpu' ou 'cuda' (défaut : cpu)."
    )
    # Booléen : on utilise BooleanOptionalAction (Python >= 3.9) plutôt
    # que type=bool. Avec type=bool, argparse convertit la chaîne passée
    # sur la ligne de commande via bool(str) — or bool("False") vaut True
    # en Python, car toute chaîne non vide est "truthy". Résultat : il
    # était impossible de désactiver l'inférence avec l'ancien script.
    # BooleanOptionalAction crée à la place deux drapeaux distincts et
    # sans ambiguïté : --lancer_inference et --no-lancer_inference.
    parser.add_argument(
        '-l', '--lancer_inference',
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Lance l'inférence avant l'évaluation. Utiliser "
             "--no-lancer_inference si l'inférence a déjà été faite "
             "précédemment sur ce dataset par ce modèle, pour recharger "
             "directement les prédictions déjà présentes sur le disque "
             "(défaut : True)."
    )

    args = parser.parse_args()

    # Lance le pipeline complet : (ré)inférence si demandé, comparaison
    # aux masques de référence (gt.json), calcul des métriques, sauvegarde
    # d'un CSV dans RESULTS et affichage d'un résumé (voir evaluator.py).
    df = evaluer_dataset_test(
        dataset_path=args.dataset_path,
        dataset_id=args.dataset_id,
        config=CONFIG,
        fold=list(args.fold),
        device=args.device,
        lancer_inference=args.lancer_inference,
        model=args.model,
    )

    # Affiche uniquement les colonnes principales du rapport ; le détail
    # complet (toutes les métriques) est déjà sauvegardé dans le CSV par
    # evaluer_dataset_test.
    print(df[["case_id", "dice", "hausdorff_95_mm", "vrai_positif", "faux_positif"]].to_string())