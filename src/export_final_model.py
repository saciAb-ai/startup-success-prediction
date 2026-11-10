"""Exporte un pipeline candidat pour l'application Streamlit (non valide).

Projet : Prediction du succes ou de l'echec des startups a partir de leurs
caracteristiques initiales (mini-projet Data Mining).

Avertissement
-------------
Ce script NE designe PAS le modele final de l'equipe et ne doit pas etre
presente comme tel. Le modele final unique sera choisi apres validation par SA
du tableau comparatif global des 6 modeles (Zero-R, KNN, Decision Tree,
Gaussian Naive Bayes, Random Forest, SVM), avec le meme CSV, le meme split et
le meme critere CV ``f1_macro``. Tant que ce tableau n'est pas valide, le
fichier produit reste un **candidat local non valide**.

Role de ce module
-----------------
- Reconstruire un pipeline candidat (Random Forest ou SVM) a partir des
  meilleurs hyperparametres enregistres par ``train_advanced.py --run``.
- L'entrainer sur ``X_train`` uniquement, puis ecrire :

  - ``models/final_pipeline.joblib`` : pipeline entraine (pretraitement +
    modele), directement chargeable par ``src/models.py`` ;
  - ``models/final_pipeline.json`` : metadonnees (modele, statut d'export,
    version de scikit-learn, variables, hyperparametres, metriques test
    enregistrees, effectifs du jeu de donnees).

- Verifier le contrat d'integration en rechargement le fichier produit avec
  ``src/models.py`` (classes ``[0, 1]`` et six colonnes d'entree).

Aucun fichier de modele n'est ecrit dans Git par ce script : c'est une
decision d'equipe (voir ``docs/app_integration.md``). Le script se contente de
produire les deux fichiers localement, et uniquement pour un essai technique.

Usage
-----
::

    python src/export_final_model.py --model random_forest
    python src/export_final_model.py --model svm
    python src/export_final_model.py --model random_forest --output /tmp/final.joblib

Le jeu de test n'est PAS utilise : les metriques de test sont relues dans
``reports/advanced_models_results.json`` (deja produites par ``--run``). Le script
n'entraine donc que sur le train, conformement au protocole.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

# Rendre les modules de src/ importables, que le fichier soit execute
# directement (python src/export_final_model.py) ou importe comme package.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import joblib
import sklearn

import models
import train_advanced as ta

DEFAULT_OUTPUT = Path("models/final_pipeline.joblib")

# Correspondance cle JSON -> nom affiche dans les metadonnees.
MODEL_NAMES: dict[str, str] = {
    "random_forest": "Random Forest",
    "svm": "SVM",
}

# Statut inscrit dans les metadonnees : le fichier n'est pas un modele final
# valide tant que SA n'a pas valide le tableau comparatif global des 6 modeles.
EXPORT_STATUS = "candidate_not_validated"
VALIDATION_REQUIRED = "tableau comparatif global des 6 modeles valide par SA"


def load_recorded_results(
    path: str | Path = ta.DEFAULT_RESULTS_PATH,
) -> dict[str, dict[str, Any]]:
    """Lit les resultats de ``--run`` (meilleurs parametres et metriques test).

    Leve
    ----
    FileNotFoundError
        Si le JSON est absent : lancer d'abord
        ``python src/train_advanced.py --run --output reports/advanced_models_results.json``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Resultats introuvables : {path}. Lancer d'abord "
            "python src/train_advanced.py --run --output reports/advanced_models_results.json"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload.get("models", {}))


def build_metadata(
    model_name: str,
    best_params: Mapping[str, Any],
    test_metrics: Mapping[str, Any],
    dataset: Mapping[str, Any],
    trained_by: str = "bl7j",
) -> dict[str, Any]:
    """Construit les metadonnees du pipeline candidat (format ``src/models.py``).

    ``test_metrics`` provient du JSON de ``--run`` : ces mesures ne sont pas
    recalculees ici, donc le jeu de test n'est pas reutilise. Les cles
    ``export_status`` et ``validation_required`` rappellent que le fichier
    produit n'est pas un modele final valide.
    """
    if model_name not in MODEL_NAMES:
        raise ValueError(
            f"Modele inconnu : {model_name!r} "
            f"(attendu : {', '.join(MODEL_NAMES)})."
        )

    # La matrice de confusion n'a pas sa place dans les metadonnees de l'app.
    metrics = {
        key: value
        for key, value in test_metrics.items()
        if key != "confusion_matrix"
    }
    return {
        "model_key": model_name,
        "model_name": MODEL_NAMES[model_name],
        "export_status": EXPORT_STATUS,
        "validation_required": VALIDATION_REQUIRED,
        "trained_by": trained_by,
        "sklearn_version": sklearn.__version__,
        "features": list(ta.FEATURE_COLUMNS),
        "best_params": {
            key.replace("model__", ""): value
            for key, value in best_params.items()
        },
        "test_metrics": metrics,
        "dataset": {
            "raw_rows": dataset.get("raw_rows"),
            "kept_rows": dataset.get("kept_rows"),
            "train_rows": dataset.get("train_rows"),
            "test_rows": dataset.get("test_rows"),
            "positive_share": dataset.get("positive_share"),
        },
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def export_pipeline(
    pipeline: Any,
    output: str | Path = DEFAULT_OUTPUT,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Ecrit le pipeline et ses metadonnees, puis verifie le contrat d'integration.

    Retour
    ------
    Path
        Chemin du fichier ``.joblib`` ecrit.

    Leve
    ----
    models.ModelContractError
        Si le pipeline recharge ne respecte pas le contrat de l'application.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output, compress=3)

    metadata_file = models.metadata_path(output)
    metadata_file.write_text(
        json.dumps(metadata or {}, indent=2, ensure_ascii=False, default=str)
        + "\n",
        encoding="utf-8",
    )

    # Rechargement reel : c'est l'application qui valide le fichier produit.
    models.load_model(output)
    return output


def run_export(
    model_name: str,
    data_path: str | Path = ta.DEFAULT_DATA_PATH,
    max_year: int = ta.DEFAULT_MAX_YEAR,
    limit: int | None = None,
    results_path: str | Path = ta.DEFAULT_RESULTS_PATH,
    output: str | Path = DEFAULT_OUTPUT,
    trained_by: str = "bl7j",
) -> dict[str, Any]:
    """Entraine un pipeline candidat sur le train et l'exporte localement.

    Le jeu de test n'est pas utilise : les metriques enregistrees dans le JSON
    de ``--run`` sont recopiees dans les metadonnees. Le fichier produit porte
    le statut ``candidate_not_validated`` : il ne remplace pas la decision
    d'equipe sur le modele final.
    """
    recorded = load_recorded_results(results_path)
    if model_name not in recorded:
        raise ValueError(
            f"Modele {model_name!r} absent de {results_path} : "
            f"modeles disponibles : {', '.join(recorded) or 'aucun'}."
        )

    entry = recorded[model_name]
    best_params = dict(entry.get("best_params", {}))
    test_metrics = dict(entry.get("test", {}))

    X_train, _X_test, y_train, _y_test, dataset = ta.load_dataset(
        path=data_path, max_year=max_year, limit=limit
    )
    pipeline = ta.build_pipeline_from_params(model_name, best_params)
    pipeline.fit(X_train, y_train)

    metadata = build_metadata(
        model_name=model_name,
        best_params=best_params,
        test_metrics=test_metrics,
        dataset=dataset,
        trained_by=trained_by,
    )
    path = export_pipeline(pipeline, output, metadata)
    return {
        "path": path,
        "metadata_path": models.metadata_path(path),
        "metadata": metadata,
        "train_rows": int(len(X_train)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Point d'entree en ligne de commande."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(MODEL_NAMES),
        help="modele candidat a exporter (le choix final appartient a l'equipe)",
    )
    parser.add_argument(
        "--data", default=str(ta.DEFAULT_DATA_PATH), help="CSV brut"
    )
    parser.add_argument(
        "--results",
        default=str(ta.DEFAULT_RESULTS_PATH),
        help="JSON produit par --run (meilleurs parametres et metriques test)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="chemin du pipeline .joblib a ecrire",
    )
    parser.add_argument("--max-year", type=int, default=ta.DEFAULT_MAX_YEAR)
    parser.add_argument("--limit", type=int, default=None, help="sous-echantillon")
    parser.add_argument(
        "--trained-by", default="bl7j", help="nom inscrit dans les metadonnees"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    exported = run_export(
        model_name=args.model,
        data_path=args.data,
        max_year=args.max_year,
        limit=args.limit,
        results_path=args.results,
        output=args.output,
        trained_by=args.trained_by,
    )

    loaded = models.load_model(exported["path"])
    prediction = models.predict_profile(
        loaded, models.make_input_frame({"city": "Paris", "founding_year": 2015})
    )
    print(f"Modele      : {loaded.name}")
    print(f"Pipeline    : {exported['path']}")
    print(f"Metadonnees : {exported['metadata_path']}")
    print(f"Statut      : {EXPORT_STATUS} (candidat, non valide par SA)")
    print(f"Entrainement: {exported['train_rows']} lignes (train uniquement)")
    print(
        f"Verification: classe={prediction.predicted_class} "
        f"score={prediction.score:.4f} type={prediction.score_kind}"
    )
    print("Contrat src/models.py verifie pour cet essai technique.")
    print(
        "Rappel : ce fichier ne designe pas le modele final ; attendre le "
        "tableau comparatif des 6 modeles valide par SA."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
