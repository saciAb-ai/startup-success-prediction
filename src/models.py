"""Chargement du pipeline final et calcul du score pour l'application Streamlit.

Projet : Prédiction du succès ou de l'échec des startups à partir de leurs
caractéristiques initiales.

Contrat d'intégration (à valider par SA et par l'équipe modèles)
----------------------------------------------------------------
- Fichier : ``models/final_pipeline.joblib``, ou le chemin donné par la
  variable d'environnement ``STARTUP_MODEL_PATH``.
- Contenu : le ``Pipeline`` scikit-learn final DÉJÀ ENTRAÎNÉ (prétraitement +
  modèle), appris sur X_train uniquement.
- Entrées : un DataFrame avec les colonnes de ``features.FEATURE_COLUMNS``.
- Classes : 0 = fermeture, 1 = IPO ou acquisition.
- Métadonnées facultatives : ``models/final_pipeline.json`` (même nom).

L'application n'entraîne aucun modèle : elle charge seulement ce fichier.

Sécurité : ``joblib.load`` peut exécuter du code. Ne charger que le fichier
produit par l'équipe, jamais un fichier fourni par un utilisateur.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

import features

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
MODEL_PATH_ENV = "STARTUP_MODEL_PATH"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "final_pipeline.joblib"
EXPECTED_CLASSES = (0, 1)


class ModelContractError(RuntimeError):
    """Le fichier chargé ne respecte pas le contrat d'intégration."""


@dataclass(frozen=True)
class LoadedModel:
    """Pipeline chargé, ses métadonnées et les remarques de chargement."""

    pipeline: Any
    path: Path
    metadata: dict = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        """Nom affiché : métadonnée ``model_name``, sinon la classe du modèle."""
        explicit = self.metadata.get("model_name")
        if explicit:
            return str(explicit)
        steps = getattr(self.pipeline, "steps", None)
        estimator = steps[-1][1] if steps else self.pipeline
        return type(estimator).__name__


@dataclass(frozen=True)
class Prediction:
    """Résultat pour un profil."""

    predicted_class: int
    score: float | None
    score_kind: str  # "probability", "decision" ou "none"


def model_path() -> Path:
    """Chemin du pipeline final (la variable d'environnement est prioritaire)."""
    override = os.environ.get(MODEL_PATH_ENV)
    return Path(override) if override else DEFAULT_MODEL_PATH


def metadata_path(path: str | Path) -> Path:
    """Fichier de métadonnées associé : même nom, extension .json."""
    return Path(path).with_suffix(".json")


def ensure_import_paths() -> None:
    """Rend importables les classes du prétraitement au chargement.

    joblib mémorise le module de chaque classe (``preprocessing._CategoricalCleaner``
    ou ``src.preprocessing._CategoricalCleaner`` selon le dossier d'où
    l'entraînement a été lancé) : les deux formes doivent être importables.
    """
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))


def load_metadata(path: str | Path) -> dict:
    """Lit les métadonnées JSON du pipeline ({} si le fichier est absent)."""
    meta_file = metadata_path(path)
    if not meta_file.is_file():
        return {}
    try:
        data = json.loads(meta_file.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelContractError(f"Métadonnées illisibles ({meta_file.name}) : {exc}") from exc
    if not isinstance(data, dict):
        raise ModelContractError(f"{meta_file.name} doit contenir un objet JSON.")
    return data


def _clean_category(value: object) -> object:
    """Texte nettoyé, ou NaN si la valeur est absente ou vide."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip()
    return text if text else np.nan


def _clean_number(value: object) -> float:
    """Nombre réel, ou NaN si la valeur est absente ou invalide."""
    if value is None:
        return np.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def make_input_frame(profile: Mapping[str, object]) -> pd.DataFrame:
    """Construit la ligne d'entrée du pipeline (colonnes de FEATURE_COLUMNS, dans l'ordre).

    Une valeur absente, None ou vide devient manquante (NaN) : le pipeline
    l'impute exactement comme pendant l'entraînement.
    """
    unknown = sorted(set(profile) - set(features.FEATURE_COLUMNS))
    if unknown:
        raise ValueError(f"Variables inconnues pour le modèle : {unknown}")
    row = {column: _clean_category(profile.get(column)) for column in features.CATEGORICAL_FEATURES}
    row.update({column: _clean_number(profile.get(column)) for column in features.NUMERIC_FEATURES})
    frame = pd.DataFrame([row], columns=list(features.FEATURE_COLUMNS))
    for column in features.CATEGORICAL_FEATURES:
        frame[column] = frame[column].astype("object")
    for column in features.NUMERIC_FEATURES:
        frame[column] = frame[column].astype("float64")
    return frame


def validate_pipeline(pipeline: Any) -> None:
    """Vérifie le contrat d'intégration ; lève ModelContractError sinon."""
    if not hasattr(pipeline, "predict"):
        raise ModelContractError("L'objet chargé n'a pas de méthode predict : ce n'est pas un modèle.")

    classes = getattr(pipeline, "classes_", None)
    if classes is None:
        raise ModelContractError("Le modèle ne semble pas entraîné (attribut classes_ absent).")
    try:
        observed = [int(value) for value in np.asarray(classes).tolist()]
    except (TypeError, ValueError):
        observed = None
    if observed != list(EXPECTED_CLASSES):
        raise ModelContractError(
            f"Classes attendues [0, 1] ; classes du modèle : {np.asarray(classes).tolist()}."
        )

    names = getattr(pipeline, "feature_names_in_", None)
    if names is not None:
        expected = set(features.FEATURE_COLUMNS)
        received = {str(name) for name in names}
        if received != expected:
            raise ModelContractError(
                "Colonnes d'entrée différentes du contrat : "
                f"manquantes {sorted(expected - received)}, en trop {sorted(received - expected)}."
            )

    try:
        prediction = np.asarray(pipeline.predict(make_input_frame({})))
    except Exception as exc:  # noqa: BLE001 - toute erreur est rapportée telle quelle
        raise ModelContractError(
            f"Le pipeline échoue sur un profil test : {type(exc).__name__} : {exc}"
        ) from exc
    if prediction.shape != (1,):
        raise ModelContractError(f"Prédiction de forme inattendue : {prediction.shape}.")


def load_model(path: str | Path | None = None) -> LoadedModel:
    """Charge le pipeline final, lit ses métadonnées et vérifie le contrat."""
    path = Path(path) if path is not None else model_path()
    if not path.is_file():
        raise FileNotFoundError(f"Modèle introuvable : {path}")

    ensure_import_paths()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline = joblib.load(path)
    notes = [
        f"Avertissement au chargement : {item.message}"
        for item in caught
        if "version" in str(item.message).lower()
    ]

    metadata = load_metadata(path)
    trained_with = metadata.get("sklearn_version")
    if trained_with and str(trained_with) != sklearn.__version__:
        notes.append(
            f"Modèle entraîné avec scikit-learn {trained_with}, version installée "
            f"{sklearn.__version__} : installer la même version est recommandé."
        )

    validate_pipeline(pipeline)
    return LoadedModel(pipeline=pipeline, path=path, metadata=metadata, notes=tuple(dict.fromkeys(notes)))


def predict_profile(model: Any, X: pd.DataFrame) -> Prediction:
    """Classe prédite et score orienté vers la classe 1, pour une seule ligne.

    Même logique que ``evaluation._positive_scores`` : ``predict_proba`` si le
    modèle le permet (estimation conditionnelle à l'échantillon d'entraînement),
    sinon ``decision_function`` (score de classement, non probabiliste).
    """
    pipeline = getattr(model, "pipeline", model)
    if len(X) != 1:
        raise ValueError("predict_profile attend exactement une ligne.")
    predicted = int(np.asarray(pipeline.predict(X))[0])
    classes = np.asarray(getattr(pipeline, "classes_", EXPECTED_CLASSES))
    positive = int(np.flatnonzero(classes == 1)[0])

    if hasattr(pipeline, "predict_proba"):
        try:
            probabilities = np.asarray(pipeline.predict_proba(X), dtype="float64")
            return Prediction(predicted, float(probabilities[0, positive]), "probability")
        except AttributeError:
            pass
    if hasattr(pipeline, "decision_function"):
        decision = float(np.asarray(pipeline.decision_function(X), dtype="float64").ravel()[0])
        return Prediction(predicted, decision if classes[-1] == 1 else -decision, "decision")
    return Prediction(predicted, None, "none")
