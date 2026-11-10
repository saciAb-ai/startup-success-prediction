"""Construit le tableau comparatif unique des six modeles.

Projet : Prediction du succes ou de l'echec des startups a partir de leurs
caracteristiques initiales (mini-projet Data Mining).

Role de ce module
-----------------
Lire les deux fichiers de resultats deja produits par l'equipe :

- ``reports/cs23_classical_results.json`` : Zero-R, KNN, Decision Tree,
  Gaussian Naive Bayes (module ``train_classical.py``) ;
- ``reports/advanced_models_results.json`` : Random Forest, SVM (module
  ``train_advanced.py``).

Puis produire une comparaison homogene :

- ``reports/models_comparison.json`` : donnees structurées et reproductibles ;
- ``reports/models_comparison.md`` : tableau lisible pour le rapport.

Le script ne reentraine aucun modele, ne lit pas le jeu de test et ne designe
pas de modele final. Il verifie seulement que les deux sources decrivent le
meme protocole (CSV, variables, split, critere CV) et signale les ecarts.

Usage
-----
::

    python src/compare_models.py
    python src/compare_models.py --json reports/models_comparison.json --md reports/models_comparison.md

Statut
------
La comparaison est informative. Le choix du modele final et sa validation
appartiennent a SA. Les metadonnees portent ``comparison_status`` pour eviter
toute ambiguite.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLASSICAL = PROJECT_ROOT / "reports" / "cs23_classical_results.json"
DEFAULT_ADVANCED = PROJECT_ROOT / "reports" / "advanced_models_results.json"
DEFAULT_JSON_OUTPUT = PROJECT_ROOT / "reports" / "models_comparison.json"
DEFAULT_MD_OUTPUT = PROJECT_ROOT / "reports" / "models_comparison.md"

COMPARISON_STATUS = "informative_not_validated"
VALIDATION_REQUIRED = "choix et validation du modele final par SA"

# Ordre du protocole commun (docs/methodology.md).
MODEL_ORDER: tuple[str, ...] = (
    "zero_r",
    "knn",
    "decision_tree",
    "naive_bayes",
    "random_forest",
    "svm",
)

MODEL_NAMES: dict[str, str] = {
    "zero_r": "Zero-R",
    "knn": "KNN",
    "decision_tree": "Decision Tree",
    "naive_bayes": "Gaussian Naive Bayes",
    "random_forest": "Random Forest",
    "svm": "SVM",
}

MODEL_GROUPS: dict[str, str] = {
    "zero_r": "classiques",
    "knn": "classiques",
    "decision_tree": "classiques",
    "naive_bayes": "classiques",
    "random_forest": "avances",
    "svm": "avances",
}

# Effectifs attendus du CSV de reference sans ISR (consigne SA).
EXPECTED_DATASET = {
    "raw_rows": 65403,
    "kept_rows": 13129,
    "train_rows": 10503,
    "test_rows": 2626,
}

METRIC_KEYS: tuple[str, ...] = (
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "roc_auc",
    "precision_fermeture",
    "recall_fermeture",
    "f1_fermeture",
)


def load_source(path: str | Path) -> dict[str, Any]:
    """Charge un fichier de resultats JSON.

    Leve
    ----
    FileNotFoundError
        Si le fichier est absent.
    ValueError
        Si la structure ne contient pas de section ``models`` exploitable.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Fichier de resultats introuvable : {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("models"), dict) or not payload["models"]:
        raise ValueError(f"Aucun modele exploitable dans : {path}")
    return payload


def normalize_path(value: object) -> str:
    """Normalise un chemin pour comparer des sources ecrites sous Windows/Linux."""
    if value is None:
        return ""
    return str(value).replace("\\", "/").strip().lstrip("./")


def _normalise_model(key: str, entry: Mapping[str, Any]) -> dict[str, Any]:
    """Extrait les champs comparables d'un modele, sans inventer de valeur."""
    test = dict(entry.get("test") or {})
    confusion = test.pop("confusion_matrix", None)
    metrics = {
        key_metric: test.get(key_metric)
        for key_metric in METRIC_KEYS
    }
    return {
        "display_name": MODEL_NAMES.get(key, key),
        "group": MODEL_GROUPS.get(key, "autre"),
        "best_params": dict(entry.get("best_params") or {}),
        "cv_f1_macro": entry.get("cv_score", entry.get("best_cv_score")),
        "cv_f1_std": entry.get("cv_std"),
        "cv_balanced_accuracy": entry.get("cv_balanced_accuracy"),
        "cv_balanced_std": entry.get("cv_balanced_std"),
        "test": metrics,
        "confusion_matrix": confusion,
        "n_train": entry.get("n_train"),
        "n_test": entry.get("n_test"),
    }


def collect_models(sources: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fusionne les modeles des sources, dans l'ordre du protocole.

    Leve
    ----
    ValueError
        Si un modele attendu manque ou si des cles inconnues apparaissent.
    """
    merged: dict[str, Any] = {}
    for source in sources:
        for key, entry in source.get("models", {}).items():
            if key in merged:
                raise ValueError(f"Modele en double : {key}")
            merged[key] = _normalise_model(key, entry)

    missing = [key for key in MODEL_ORDER if key not in merged]
    unexpected = sorted(set(merged) - set(MODEL_ORDER))
    if missing:
        raise ValueError(f"Modeles manquants : {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"Modeles inattendus : {', '.join(unexpected)}")
    return {key: merged[key] for key in MODEL_ORDER}


def _check(name: str, expected: Any, values: Sequence[Any]) -> dict[str, Any]:
    """Construit un controle de coherence lisible."""
    normalised = [value if not isinstance(value, str) else normalize_path(value) for value in values]
    expected_norm = expected if not isinstance(expected, str) else normalize_path(expected)
    ok = all(value == expected_norm for value in normalised)
    return {
        "name": name,
        "expected": expected_norm,
        "values": normalised,
        "ok": ok,
    }


def protocol_checks(
    classical: Mapping[str, Any], advanced: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Verifie que les deux sources decrivent le meme protocole."""
    c_proto = dict(classical.get("protocol") or {})
    a_proto = dict(advanced.get("protocol") or {})
    c_data = dict(classical.get("dataset") or {})
    a_data = dict(advanced.get("dataset") or {})

    checks = [
        _check(
            "chemin_donnees",
            "data/raw/startup_data.csv",
            [c_proto.get("data_path"), a_proto.get("data_path")],
        ),
        _check(
            "variables",
            list(a_proto.get("features") or []),
            [c_proto.get("features"), a_proto.get("features")],
        ),
        _check(
            "exclues",
            ["founding_month"],
            [c_proto.get("excluded"), a_proto.get("excluded")],
        ),
        _check(
            "critere_cv",
            "f1_macro",
            [c_proto.get("scoring"), a_proto.get("scoring")],
        ),
        _check("folds", 5, [c_proto.get("cv_folds"), a_proto.get("cv_folds")]),
        _check(
            "graine",
            42,
            [c_proto.get("random_state"), a_proto.get("random_state")],
        ),
    ]

    for name, expected in EXPECTED_DATASET.items():
        checks.append(
            _check(
                f"effectif_{name}",
                expected,
                [c_data.get(name), a_data.get(name)],
            )
        )

    return checks


def build_comparison(
    classical_path: str | Path = DEFAULT_CLASSICAL,
    advanced_path: str | Path = DEFAULT_ADVANCED,
) -> dict[str, Any]:
    """Charge les deux sources et construit la comparaison complete."""
    classical = load_source(classical_path)
    advanced = load_source(advanced_path)

    classical_versions = dict(classical.get("versions") or {})
    advanced_versions = dict(advanced.get("versions") or {})

    checks = protocol_checks(classical, advanced)
    return {
        "comparison_status": COMPARISON_STATUS,
        "validation_required": VALIDATION_REQUIRED,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "src/compare_models.py",
        "source_files": {
            "classical": str(classical_path),
            "advanced": str(advanced_path),
        },
        "protocol": {
            "data_path": "data/raw/startup_data.csv",
            "features": list((advanced.get("protocol") or {}).get("features") or []),
            "excluded": ["founding_month"],
            "scoring": "f1_macro",
            "cv_folds": 5,
            "random_state": 42,
        },
        "dataset": {
            key: value
            for key, value in {
                **EXPECTED_DATASET,
                "positive_share_train": (classical.get("dataset") or {}).get(
                    "positive_share_train"
                ),
            }.items()
        },
        "protocol_checks": checks,
        "protocol_ok": all(check["ok"] for check in checks),
        "versions": {
            "classical": classical_versions,
            "advanced": advanced_versions,
            "versions_comparable": bool(
                classical_versions
                and advanced_versions
                and all(
                    classical_versions.get(key) == advanced_versions.get(key)
                    for key in ("scikit-learn", "joblib", "pandas", "numpy", "python")
                )
            ),
        },
        "models": collect_models([classical, advanced]),
    }


def _fmt(value: Any, digits: int = 4) -> str:
    """Formate un nombre pour le Markdown (virgule decimale, tiret si absent)."""
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}".replace(".", ",")
    except (TypeError, ValueError):
        return str(value)


def _versions_row(versions: Mapping[str, Any], key: str) -> str:
    return str(versions.get(key, "—"))


def format_markdown(comparison: Mapping[str, Any]) -> str:
    """Rend la comparaison en Markdown pret a coller dans le rapport."""
    lines: list[str] = []
    add = lines.append

    add("# Comparaison des six modeles — protocole commun")
    add("")
    add(
        "> **Statut : comparaison informative, non validee.** Le modele final "
        "n'est pas designe ici : le choix et la validation appartiennent a SA."
    )
    add("")
    add(f"> Genere par `src/compare_models.py` le {comparison['generated_at']}.")
    add("")

    add("## 1. Protocole")
    add("")
    add("| Element | Valeur |")
    add("|---|---|")
    proto = comparison["protocol"]
    add(f"| CSV | `{proto['data_path']}` |")
    add(f"| Variables | {', '.join(proto['features'])} |")
    add(f"| Exclues | {', '.join(proto['excluded'])} |")
    add(f"| Critere CV | `{proto['scoring']}` |")
    add(f"| Folds | {proto['cv_folds']} |")
    add(f"| Graine | {proto['random_state']} |")
    add("")

    dataset = comparison["dataset"]
    add("| Effectif | Valeur |")
    add("|---|---|")
    add(f"| Lignes brutes | {dataset.get('raw_rows')} |")
    add(f"| Lignes retenues | {dataset.get('kept_rows')} |")
    add(f"| Train / test | {dataset.get('train_rows')} / {dataset.get('test_rows')} |")
    if dataset.get("positive_share_train") is not None:
        add(f"| Part classe 1 (train) | {_fmt(dataset['positive_share_train'])} |")
    add("")

    checks = comparison["protocol_checks"]
    if not all(check["ok"] for check in checks):
        add("### Controles de coherence")
        add("")
        add("| Controle | Attendu | Valeurs | Statut |")
        add("|---|---|---|---|")
        for check in checks:
            status = "OK" if check["ok"] else "ECART"
            values = ", ".join(str(value) for value in check["values"])
            add(f"| {check['name']} | {check['expected']} | {values} | {status} |")
        add("")

    add("## 2. Performances")
    add("")
    add(
        "| Modele | Famille | F1 macro CV | F1 macro test | ROC-AUC | Accuracy | "
        "Accuracy equilibree | Precision macro | Rappel macro |"
    )
    add("|---|---|---|---|---|---|---|---|---|")
    for key, model in comparison["models"].items():
        test = model["test"]
        add(
            f"| {model['display_name']} | {model['group']} | "
            f"{_fmt(model['cv_f1_macro'])} | {_fmt(test.get('f1_macro'))} | "
            f"{_fmt(test.get('roc_auc'))} | {_fmt(test.get('accuracy'))} | "
            f"{_fmt(test.get('balanced_accuracy'))} | "
            f"{_fmt(test.get('precision_macro'))} | "
            f"{_fmt(test.get('recall_macro'))} |"
        )
    add("")

    add("### Metriques de la classe 0 (fermeture)")
    add("")
    add("| Modele | Precision | Rappel | F1 |")
    add("|---|---|---|---|")
    for model in comparison["models"].values():
        test = model["test"]
        add(
            f"| {model['display_name']} | {_fmt(test.get('precision_fermeture'))} | "
            f"{_fmt(test.get('recall_fermeture'))} | {_fmt(test.get('f1_fermeture'))} |"
        )
    add("")

    add("### Matrices de confusion (lignes = reel, colonnes = predit)")
    add("")
    add("| Modele | 0 / 0 | 0 / 1 | 1 / 0 | 1 / 1 |")
    add("|---|---|---|---|---|")
    for model in comparison["models"].values():
        matrix = model.get("confusion_matrix") or [[None] * 2] * 2
        flat = [matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]]
        add(f"| {model['display_name']} | " + " | ".join(_fmt(v, 0) for v in flat) + " |")
    add("")

    add("## 3. Versions")
    add("")
    add("| Bibliotheque | Classiques | Avances |")
    add("|---|---|---|")
    c_versions = comparison["versions"]["classical"]
    a_versions = comparison["versions"]["advanced"]
    for key in ("python", "numpy", "pandas", "scikit-learn", "joblib"):
        add(
            f"| {key} | {_versions_row(c_versions, key)} | "
            f"{_versions_row(a_versions, key)} |"
        )
    add("")
    if not comparison["versions"]["versions_comparable"]:
        add(
            "> ⚠️ Les deux sources n'ont pas ete produites avec les memes versions "
            "de bibliotheques (ou les versions avancees ne sont pas enregistrees). "
            "Pour une comparaison stricte, rejouer les deux modules dans le meme "
            "environnement avant la decision finale."
        )
        add("")

    add("## 4. Lecture")
    add("")
    add(
        "Ce tableau ne tranche pas : il rassemble les six modeles sur le meme "
        "CSV, le meme split et le meme critere CV `f1_macro`. La decision du "
        "modele final revient a SA, apres verification des versions et du "
        "protocole."
    )
    add("")
    add("Regeneration :")
    add("")
    add("```bash")
    add("python src/compare_models.py")
    add("```")
    add("")
    return "\n".join(lines)


def write_reports(
    comparison: Mapping[str, Any],
    json_path: str | Path = DEFAULT_JSON_OUTPUT,
    md_path: str | Path = DEFAULT_MD_OUTPUT,
) -> tuple[Path, Path]:
    """Ecrit le JSON et le Markdown, puis retourne les deux chemins."""
    json_path = Path(json_path)
    md_path = Path(md_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(format_markdown(comparison), encoding="utf-8")
    return json_path, md_path


def main(argv: Sequence[str] | None = None) -> int:
    """Point d'entree en ligne de commande."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--classical",
        default=str(DEFAULT_CLASSICAL),
        help="JSON des modeles classiques (CS23)",
    )
    parser.add_argument(
        "--advanced",
        default=str(DEFAULT_ADVANCED),
        help="JSON des modeles avances",
    )
    parser.add_argument(
        "--json", default=str(DEFAULT_JSON_OUTPUT), help="sortie JSON"
    )
    parser.add_argument("--md", default=str(DEFAULT_MD_OUTPUT), help="sortie Markdown")
    args = parser.parse_args(list(argv) if argv is not None else None)

    comparison = build_comparison(args.classical, args.advanced)
    json_path, md_path = write_reports(comparison, args.json, args.md)

    models = comparison["models"]
    print(f"Modeles compares : {len(models)}")
    for key, model in models.items():
        print(
            f"  {model['display_name']:<22} "
            f"F1 macro test={_fmt(model['test'].get('f1_macro'))}"
        )
    print(f"Protocole coherent : {'oui' if comparison['protocol_ok'] else 'NON'}")
    print(f"Versions comparables : {'oui' if comparison['versions']['versions_comparable'] else 'NON'}")
    print(f"JSON : {json_path}")
    print(f"Markdown : {md_path}")
    print(
        "Statut : comparaison informative, non validee. "
        "Le modele final reste a valider par SA."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
