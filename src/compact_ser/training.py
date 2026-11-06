"""Training and validation loops shared by teachers and students."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from compact_ser.audio_data import EncodedUtterance, SpeechBatch
from compact_ser.metrics import (
    expected_calibration_error,
    macro_f1_score,
    negative_log_likelihood,
    unweighted_average_recall,
)
from compact_ser.models import SpeechEmotionModel
from compact_ser.study_design import TrainingSettings
from compact_ser.text_processing import TranscriptTokenizer, word_error_rate

LossTerms = dict[str, torch.Tensor]
Diagnostics = dict[str, float]
LossFunction = Callable[[SpeechBatch], tuple[LossTerms, Diagnostics]]
Validator = Callable[[nn.Module], dict[str, float]]


@dataclass
class TrainingOutcome:
    """Result of a training stage.

    Attributes:
        best_state: Model weights from the update with the best validation UAR, on the CPU.
        best_update: The update that produced those weights.
        best_metrics: Validation metrics at that update.
        validation_history: Validation metrics from every validation pass.
        gradient_history: Encoder gradient shares and diagnostics from each logged update.
    """

    best_state: dict[str, torch.Tensor] | None = None
    best_update: int | None = None
    best_metrics: dict[str, float] = field(default_factory=dict)
    validation_history: list[dict[str, float]] = field(default_factory=list)
    gradient_history: list[dict[str, float]] = field(default_factory=list)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators.

    Args:
        seed: The seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate_model(
    model: SpeechEmotionModel,
    items: Sequence[EncodedUtterance],
    num_emotions: int,
    device: torch.device,
    tokenizer: TranscriptTokenizer | None = None,
) -> dict[str, Any]:
    """Evaluate a model one utterance at a time.

    Utterances are not batched because padding changes the predictions of models with group
    normalization.

    Args:
        model: The model to evaluate.
        items: Utterances to score.
        num_emotions: Number of emotion classes.
        device: Device used for inference.
        tokenizer: Tokenizer used to report the word error rate, if the model has a transcript head.

    Returns:
        Metrics, class probabilities, and true labels.
    """
    was_training = model.training
    model.eval()
    probabilities = []
    references = []
    hypotheses = []
    for item in items:
        waveform = torch.from_numpy(item.waveform).unsqueeze(0).to(device)
        output = model(waveform, torch.ones_like(waveform, dtype=torch.long))
        probabilities.append(output.emotion_logits.softmax(dim=-1)[0].cpu().numpy())
        if tokenizer is not None and output.transcript_logits is not None:
            references.append(item.text)
            hypotheses.append(tokenizer.decode_ctc(output.transcript_logits[0].argmax(-1).tolist()))
    model.train(was_training)

    probability_matrix = np.stack(probabilities)
    labels = np.array([item.emotion for item in items])
    predictions = probability_matrix.argmax(axis=1)
    metrics = {
        "uar": unweighted_average_recall(labels, predictions, num_emotions),
        "macro_f1": macro_f1_score(labels, predictions, num_emotions),
        "negative_log_likelihood": negative_log_likelihood(probability_matrix, labels),
        "expected_calibration_error": expected_calibration_error(probability_matrix, labels),
    }
    if references:
        metrics["word_error_rate"] = word_error_rate(references, hypotheses)
    return {"metrics": metrics, "probabilities": probability_matrix, "labels": labels}


def compute_gradient_shares(
    terms: LossTerms, parameters: Sequence[torch.Tensor]
) -> dict[str, float]:
    """Measure each loss term's share of the gradient norm on the given parameters.

    Args:
        terms: Weighted loss terms that are still attached to the graph.
        parameters: Parameters shared by the terms.

    Returns:
        Each term's share of the summed gradient norms.
    """
    norms = {}
    for name, term in terms.items():
        gradients = torch.autograd.grad(term, parameters, retain_graph=True, allow_unused=True)
        squared = sum(gradient.pow(2).sum() for gradient in gradients if gradient is not None)
        norms[name] = float(torch.as_tensor(squared).sqrt())
    total = sum(norms.values())
    return {f"{name}_gradient_share": norm / max(total, 1e-12) for name, norm in norms.items()}


def train_stage(
    model: SpeechEmotionModel,
    compute_loss: LossFunction,
    batches: Iterator[SpeechBatch],
    updates: int,
    settings: TrainingSettings,
    validate: Validator | None = None,
) -> TrainingOutcome:
    """Run a fixed number of updates, keeping the weights with the best validation UAR.

    Args:
        model: The model to train.
        compute_loss: Returns the loss terms and diagnostics for a batch.
        batches: Training batches.
        updates: Number of optimizer updates.
        settings: Optimization settings.
        validate: Returns validation metrics, including `uar`. If omitted, no weights are kept.

    Returns:
        The best weights and training records.
    """
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    encoder_layers = [
        parameter for parameter in model.encoder.encoder.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(parameters, lr=settings.learning_rate)
    outcome = TrainingOutcome()
    model.train()
    for update in range(1, updates + 1):
        optimizer.zero_grad(set_to_none=True)
        terms, diagnostics = compute_loss(next(batches))
        if update == 1 or update % settings.gradient_log_interval == 0:
            shares = compute_gradient_shares(terms, encoder_layers)
            outcome.gradient_history.append({"update": update, **shares, **diagnostics})
        sum(terms.values()).backward()
        torch.nn.utils.clip_grad_norm_(parameters, settings.max_gradient_norm)
        optimizer.step()

        is_validation_update = update % settings.validation_interval == 0 or update == updates
        if validate is not None and is_validation_update:
            metrics = validate(model)
            outcome.validation_history.append({"update": update, **metrics})
            if outcome.best_state is None or metrics["uar"] > outcome.best_metrics["uar"]:
                outcome.best_state = {
                    name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                }
                outcome.best_update = update
                outcome.best_metrics = metrics
    return outcome
