"""Classification and calibration metrics."""

from __future__ import annotations

import numpy as np


def compute_confusion_matrix(
    labels: np.ndarray, predictions: np.ndarray, num_classes: int
) -> np.ndarray:
    """Count predictions for each true class.

    Args:
        labels: True class indices.
        predictions: Predicted class indices.
        num_classes: Number of classes.

    Returns:
        A matrix with true classes in rows and predicted classes in columns.
    """
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    return matrix


def unweighted_average_recall(
    labels: np.ndarray, predictions: np.ndarray, num_classes: int
) -> float:
    """Average the recall across classes that occur in the labels.

    Args:
        labels: True class indices.
        predictions: Predicted class indices.
        num_classes: Number of classes.

    Returns:
        The unweighted average recall.
    """
    matrix = compute_confusion_matrix(labels, predictions, num_classes)
    present = matrix.sum(axis=1) > 0
    return float((matrix.diagonal()[present] / matrix.sum(axis=1)[present]).mean())


def macro_f1_score(labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> float:
    """Average the F1 score across classes that occur in the labels.

    Args:
        labels: True class indices.
        predictions: Predicted class indices.
        num_classes: Number of classes.

    Returns:
        The macro-averaged F1 score.
    """
    matrix = compute_confusion_matrix(labels, predictions, num_classes)
    present = matrix.sum(axis=1) > 0
    correct = matrix.diagonal().astype(float)
    denominator = matrix.sum(axis=1) + matrix.sum(axis=0)
    scores = np.divide(2 * correct, denominator, out=np.zeros(num_classes), where=denominator > 0)
    return float(scores[present].mean())


def negative_log_likelihood(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Compute the mean negative log-likelihood of the true classes.

    Args:
        probabilities: Predicted probabilities with shape `(items, classes)`.
        labels: True class indices.

    Returns:
        The mean negative log-likelihood.
    """
    selected = probabilities[np.arange(labels.size), labels]
    return float(-np.log(np.clip(selected, 1e-12, 1.0)).mean())


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, num_bins: int = 10
) -> float:
    """Compute the expected calibration error with equal-width confidence bins.

    Args:
        probabilities: Predicted probabilities with shape `(items, classes)`.
        labels: True class indices.
        num_bins: Number of confidence bins.

    Returns:
        The weighted mean gap between confidence and accuracy.
    """
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    bins = np.minimum((confidence * num_bins).astype(int), num_bins - 1)
    error = 0.0
    for index in range(num_bins):
        members = bins == index
        if members.any():
            gap = abs(float(correct[members].mean()) - float(confidence[members].mean()))
            error += float(members.mean()) * gap
    return error
