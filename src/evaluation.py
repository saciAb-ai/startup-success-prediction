"""Mesures communes d'evaluation des classifieurs binaires.

Projet : Prédiction du succès ou de l'échec des startups à partir de leurs caractéristiques initiales.
Classe 0 = fermeture (closed) ; classe 1 = acquisition ou IPO.

Toutes les mesures doivent etre calculees sur le meme jeu de test final,
une fois le choix des hyperparametres termine sur le jeu d'entrainement.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _positive_scores(model, X_test):
    """Score oriente vers la classe 1 ; None si indisponible.

    predict_proba retourne une estimation de probabilite conditionnelle au
    dataset d'entrainement (pas une probabilite reelle de succes futur).
    decision_function retourne un score de classement, pas une probabilite.
    """
    classes = np.asarray(getattr(model, "classes_", []))
    if len(classes) != 2 or 1 not in classes:
        return None
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X_test))
        if probabilities.ndim == 2 and probabilities.shape[1] == 2:
            return probabilities[:, int(np.flatnonzero(classes == 1)[0])]
    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(X_test))
        if decision.ndim == 1:
            # sklearn oriente les scores binaires vers classes_[1].
            return decision if classes[1] == 1 else -decision
    return None


def evaluate_classifier(model, X_test, y_test) -> dict:
    """Calcule un rapport commun pour tout classifieur scikit-learn entraine.

    Le modele doit exposer predict ; predict_proba ou decision_function sont
    facultatifs (ROC-AUC = None si aucun score n'est disponible).
    """
    y_true = np.asarray(y_test)
    if y_true.ndim != 1 or len(y_true) == 0:
        raise ValueError("y_test doit etre un vecteur non vide.")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_test doit contenir uniquement les etiquettes 0 et 1.")

    y_pred = np.asarray(model.predict(X_test))
    if y_pred.shape != y_true.shape or not np.isin(y_pred, [0, 1]).all():
        raise ValueError("Le classifieur doit predire uniquement les classes 0 et 1.")

    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_fermeture": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_fermeture": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_fermeture": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "roc_auc": None,
    }

    scores = _positive_scores(model, X_test)
    if scores is not None and len(set(y_true.tolist())) == 2:
        scores = np.asarray(scores)
        if scores.shape == y_true.shape and np.isfinite(scores).all():
            result["roc_auc"] = float(roc_auc_score(y_true, scores))
    return result
