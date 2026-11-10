"""Tests des metriques d'evaluation (``src/evaluation.py``).

Contribution CS30. Les valeurs attendues sont calculees a la main sur de tres
petits jeux, puis comparees au rapport produit par ``evaluate_classifier``.
Les tests verifient aussi l'orientation du score vers la classe 1, un point
critique pour l'interpretation du ROC-AUC.
"""

from __future__ import annotations

import numpy as np
import pytest

import evaluation


class _PredictOnly:
    """Classifieur minimal : uniquement ``predict`` et ``classes_``."""

    def __init__(self, y_pred, classes=(0, 1)):
        self.classes_ = np.asarray(classes)
        self.y_pred = np.asarray(y_pred)

    def predict(self, X):
        return self.y_pred


class _ProbaModel(_PredictOnly):
    """Classifieur avec ``predict_proba`` aligne sur l'ordre de ``classes_``."""

    def __init__(self, y_pred, p_class1, classes=(0, 1)):
        super().__init__(y_pred, classes)
        probability = np.asarray(p_class1, dtype=float)
        columns = [probability if label == 1 else 1.0 - probability for label in self.classes_]
        self.proba = np.column_stack(columns)

    def predict_proba(self, X):
        return self.proba


class _DecisionModel(_PredictOnly):
    """Classifieur avec ``decision_function`` (pas de probabilites)."""

    def __init__(self, y_pred, decision, classes=(0, 1)):
        super().__init__(y_pred, classes)
        self.decision = np.asarray(decision, dtype=float)

    def decision_function(self, X):
        return self.decision


def _X(n_rows: int):
    """Entree factice : les stubs ignorent son contenu."""
    return np.zeros((n_rows, 1))


# ---------------------------------------------------------------------------
# Exemple connu, calcule a la main
# ---------------------------------------------------------------------------


def test_known_confusion_matrix_and_metrics() -> None:
    """y = [0,0,0,1,1], pred = [0,0,1,1,1] : TN=2, FP=1, FN=0, TP=2."""
    y_true = [0, 0, 0, 1, 1]
    y_pred = [0, 0, 1, 1, 1]
    result = evaluation.evaluate_classifier(_PredictOnly(y_pred), _X(5), y_true)

    assert result["confusion_matrix"] == [[2, 1], [0, 2]]
    assert result["accuracy"] == pytest.approx(4 / 5)
    assert result["balanced_accuracy"] == pytest.approx((2 / 3 + 1) / 2)
    assert result["precision_macro"] == pytest.approx((1 + 2 / 3) / 2)
    assert result["recall_macro"] == pytest.approx((2 / 3 + 1) / 2)
    assert result["f1_macro"] == pytest.approx(0.8)
    assert result["precision_fermeture"] == pytest.approx(1.0)
    assert result["recall_fermeture"] == pytest.approx(2 / 3)
    assert result["f1_fermeture"] == pytest.approx(0.8)


def test_balanced_two_by_two_example() -> None:
    """Un cas symetriquement incorrect : toutes les mesures valent 0,5."""
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 0]
    result = evaluation.evaluate_classifier(_PredictOnly(y_pred), _X(4), y_true)

    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["balanced_accuracy"] == pytest.approx(0.5)
    assert result["precision_macro"] == pytest.approx(0.5)
    assert result["recall_macro"] == pytest.approx(0.5)
    assert result["f1_macro"] == pytest.approx(0.5)


def test_confusion_matrix_uses_zero_one_order() -> None:
    """Ligne 0 = fermeture, ligne 1 = succes ; colonnes dans le meme ordre."""
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 0, 1]
    result = evaluation.evaluate_classifier(_PredictOnly(y_pred), _X(4), y_true)

    assert result["confusion_matrix"] == [[1, 1], [1, 1]]


def test_result_types_are_json_friendly() -> None:
    """Les valeurs sont des flottants et la matrice une liste de listes."""
    result = evaluation.evaluate_classifier(_PredictOnly([0, 1, 0, 1]), _X(4), [0, 1, 0, 1])

    for key in (
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_fermeture",
        "recall_fermeture",
        "f1_fermeture",
    ):
        assert isinstance(result[key], float)
    assert isinstance(result["confusion_matrix"], list)
    assert all(isinstance(row, list) for row in result["confusion_matrix"])


# ---------------------------------------------------------------------------
# ROC-AUC : orientation vers la classe 1
# ---------------------------------------------------------------------------


def test_roc_auc_uses_the_probability_of_class_one() -> None:
    """Un score parfaitement croissant avec la classe 1 donne un AUC de 1."""
    y_true = [0, 0, 1, 1]
    result = evaluation.evaluate_classifier(
        _ProbaModel(y_true, p_class1=[0.1, 0.2, 0.8, 0.9]),
        _X(4),
        y_true,
    )

    assert result["roc_auc"] == pytest.approx(1.0)


def test_roc_auc_is_inverted_when_probabilities_are_inverted() -> None:
    """Un score anticorrele a la classe 1 donne un AUC de 0, pas de 1."""
    y_true = [0, 0, 1, 1]
    result = evaluation.evaluate_classifier(
        _ProbaModel(y_true, p_class1=[0.9, 0.8, 0.2, 0.1]),
        _X(4),
        y_true,
    )

    assert result["roc_auc"] == pytest.approx(0.0)


def test_roc_auc_handles_reversed_class_order() -> None:
    """L'ordre de ``classes_`` ne doit pas inverser silencieusement l'AUC."""
    y_true = [0, 0, 1, 1]
    result = evaluation.evaluate_classifier(
        _ProbaModel(y_true, p_class1=[0.1, 0.2, 0.8, 0.9], classes=(1, 0)),
        _X(4),
        y_true,
    )

    assert result["roc_auc"] == pytest.approx(1.0)


def test_roc_auc_uses_decision_function_orientation() -> None:
    """Sans probabilites, le score de decision doit rester oriente vers la classe 1."""
    y_true = [0, 0, 1, 1]
    result = evaluation.evaluate_classifier(
        _DecisionModel(y_true, decision=[-2.0, -1.0, 1.0, 2.0]),
        _X(4),
        y_true,
    )

    assert result["roc_auc"] == pytest.approx(1.0)


def test_roc_auc_handles_reversed_decision_classes() -> None:
    """Avec ``classes_ = (1, 0)``, la decision positive pointe vers la classe 0."""
    y_true = [0, 0, 1, 1]
    result = evaluation.evaluate_classifier(
        _DecisionModel(y_true, decision=[2.0, 1.0, -1.0, -2.0], classes=(1, 0)),
        _X(4),
        y_true,
    )

    assert result["roc_auc"] == pytest.approx(1.0)


def test_roc_auc_is_none_without_scores() -> None:
    """Un modele qui ne fournit ni probabilite ni score n'a pas d'AUC."""
    result = evaluation.evaluate_classifier(_PredictOnly([0, 1, 0, 1]), _X(4), [0, 1, 0, 1])

    assert result["roc_auc"] is None


@pytest.mark.filterwarnings("ignore:A single label was found")
def test_roc_auc_is_none_when_test_has_a_single_class() -> None:
    """L'AUC n'est pas definie quand une seule classe est presente."""
    result = evaluation.evaluate_classifier(
        _ProbaModel([0, 0, 0], p_class1=[0.1, 0.2, 0.3]),
        _X(3),
        [0, 0, 0],
    )

    assert result["roc_auc"] is None
    assert result["accuracy"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Validation des entrees
# ---------------------------------------------------------------------------


def test_rejects_empty_y_test() -> None:
    """Un jeu de test vide ne peut pas etre evalue."""
    with pytest.raises(ValueError, match="non vide"):
        evaluation.evaluate_classifier(_PredictOnly([]), _X(0), [])


def test_rejects_non_binary_y_test() -> None:
    """Les etiquettes attendues sont uniquement 0 et 1."""
    with pytest.raises(ValueError, match="0 et 1"):
        evaluation.evaluate_classifier(_PredictOnly([0, 1]), _X(2), [0, 2])


def test_rejects_two_dimensional_y_test() -> None:
    """``y_test`` doit etre un vecteur."""
    with pytest.raises(ValueError, match="vecteur"):
        evaluation.evaluate_classifier(_PredictOnly([0, 1]), _X(2), [[0], [1]])


def test_rejects_non_binary_predictions() -> None:
    """Un classifieur qui predit une autre classe est invalide."""
    with pytest.raises(ValueError, match="classes 0 et 1"):
        evaluation.evaluate_classifier(_PredictOnly([0, 2]), _X(2), [0, 1])


def test_rejects_prediction_length_mismatch() -> None:
    """Les predictions doivent couvrir exactement le jeu de test."""
    with pytest.raises(ValueError, match="classes 0 et 1"):
        evaluation.evaluate_classifier(_PredictOnly([0, 1, 0]), _X(2), [0, 1])
