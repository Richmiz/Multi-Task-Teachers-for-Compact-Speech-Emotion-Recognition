"""Training and validation loops shared by teachers and students."""

from __future__ import annotations

import random
import time
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
    validation_seconds: float = 0.0
    peak_gpu_memory_gib: float | None = None


def precision_context(device: torch.device, mixed_precision: bool) -> torch.autocast:
    """Return a context that computes in bfloat16 on a GPU when mixed precision is enabled.

    Operations that need full precision, such as losses and normalization, stay in float32
    under PyTorch's autocast rules. On a CPU the context does nothing.
    """
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=mixed_precision and device.type == "cuda",
    )


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
    mixed_precision: bool = False,
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
        mixed_precision: Whether to compute in bfloat16 on a GPU, as in training.

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
        with precision_context(device, mixed_precision):
            output = model(waveform, torch.ones_like(waveform, dtype=torch.long))
        probabilities.append(output.emotion_logits.float().softmax(dim=-1)[0].cpu().numpy())
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


def head_only_updates(updates: int, settings: TrainingSettings) -> int:
    """Return how many of a stage's first updates train only the task heads."""
    return round(settings.head_only_share * updates)


def learning_rate_factor(update: int, updates: int, settings: TrainingSettings) -> float:
    """Return the share of the peak learning rate used at an update.

    The heads-only updates use the peak rate. Over the remaining updates the rate warms up
    linearly, holds, and then decays linearly to zero, following wav2vec 2.0's schedule.

    Args:
        update: The update, counted from 1.
        updates: Number of updates in the stage.
        settings: Optimization settings.

    Returns:
        A factor between 0 and 1.
    """
    heads_only = head_only_updates(updates, settings)
    if update <= heads_only:
        return 1.0
    position = update - heads_only
    span = updates - heads_only
    warmup = settings.warmup_share * span
    hold = settings.hold_share * span
    if position <= warmup:
        return position / warmup
    if position <= warmup + hold:
        return 1.0
    return max(0.0, (span - position) / (span - warmup - hold))


def _position(update: int, updates: int, updates_per_epoch: int | None) -> str:
    position = f"update {update}/{updates}"
    if updates_per_epoch:
        position += f", epoch {update / updates_per_epoch:.1f}"
    return position


def _duration(seconds: float) -> str:
    minutes = round(seconds / 60)
    return f"{minutes // 60} h {minutes % 60:02d} min"


def _report_validation(position: str, metrics: dict[str, float], heads_only: bool) -> None:
    line = f"  {position}: validation UAR {metrics['uar']:.3f}"
    if "word_error_rate" in metrics:
        line += f", WER {metrics['word_error_rate']:.3f}"
    if heads_only:
        line += " (heads only)"
    print(line, flush=True)


def _report_progress(
    position: str,
    loss_sums: dict[str, torch.Tensor],
    interval_updates: int,
    seconds_per_update: float,
    remaining_updates: int,
    learning_rate: float,
    heads_only: bool,
) -> None:
    losses = {name: float(total) / interval_updates for name, total in loss_sums.items()}
    parts = ", ".join(f"{name} {value:.3f}" for name, value in losses.items())
    line = (
        f"  {position}: loss {sum(losses.values()):.3f} ({parts}), "
        f"learning rate {learning_rate:.1e}, {seconds_per_update:.2f} s/update, "
        f"about {_duration(remaining_updates * seconds_per_update)} left"
    )
    if heads_only:
        line += " (heads only)"
    print(line, flush=True)


def train_stage(
    model: SpeechEmotionModel,
    compute_loss: LossFunction,
    batches: Iterator[SpeechBatch],
    updates: int,
    settings: TrainingSettings,
    validate: Validator | None = None,
    on_encoder_start: Callable[[], None] | None = None,
    updates_per_epoch: int | None = None,
) -> TrainingOutcome:
    """Run a fixed number of updates, keeping the weights with the best validation UAR.

    The first updates train only the task heads, with the encoder frozen, so that the heads'
    random initial gradients do not disturb the pretrained encoder. The encoder then trains with a
    learning rate that warms up, holds, and decays.

    Args:
        model: The model to train.
        compute_loss: Returns the loss terms and diagnostics for a batch.
        batches: Training batches.
        updates: Number of optimizer updates.
        settings: Optimization settings.
        validate: Returns validation metrics, including `uar`. If omitted, no weights are kept.
        on_encoder_start: Called once, just before the encoder starts training, for example to
            set loss weights from the trained heads.
        updates_per_epoch: Updates in one pass through the training data, used in the log.

    Returns:
        The best weights and training records.
    """
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    encoder_parameters = [
        parameter for parameter in model.encoder.parameters() if parameter.requires_grad
    ]
    encoder_layers = [
        parameter for parameter in model.encoder.encoder.parameters() if parameter.requires_grad
    ]
    heads_only = head_only_updates(updates, settings)
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(parameters, lr=settings.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: learning_rate_factor(step + 1, updates, settings)
    )
    outcome = TrainingOutcome()
    model.train()
    for parameter in encoder_parameters:
        parameter.requires_grad_(heads_only == 0)
    if updates_per_epoch:
        epochs = updates / updates_per_epoch
        print(
            f"  {updates} updates of {settings.batch_size} utterances: {updates_per_epoch} per "
            f"epoch, about {epochs:.0f} epochs, the first {heads_only} training heads only.",
            flush=True,
        )
    loss_sums: dict[str, torch.Tensor] = {}
    interval_updates = 0
    interval_started = time.perf_counter()
    for update in range(1, updates + 1):
        if update == heads_only + 1:
            for parameter in encoder_parameters:
                parameter.requires_grad_(True)
            if on_encoder_start is not None:
                on_encoder_start()
            if heads_only:
                position = _position(update, updates, updates_per_epoch)
                print(f"  {position}: encoder training starts.", flush=True)
        encoder_trains = update > heads_only
        optimizer.zero_grad(set_to_none=True)
        with precision_context(device, settings.mixed_precision):
            terms, diagnostics = compute_loss(next(batches))
        is_log_update = update == heads_only + 1 or update % settings.gradient_log_interval == 0
        if encoder_trains and is_log_update:
            shares = compute_gradient_shares(terms, encoder_layers)
            outcome.gradient_history.append({"update": update, **shares, **diagnostics})
        sum(terms.values()).backward()
        torch.nn.utils.clip_grad_norm_(parameters, settings.max_gradient_norm)
        learning_rate = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        for name, term in terms.items():
            loss_sums[name] = loss_sums.get(name, 0.0) + term.detach()
        interval_updates += 1
        if update % settings.gradient_log_interval == 0 or update == updates:
            seconds_per_update = (time.perf_counter() - interval_started) / interval_updates
            _report_progress(
                _position(update, updates, updates_per_epoch),
                loss_sums,
                interval_updates,
                seconds_per_update,
                updates - update,
                learning_rate,
                not encoder_trains,
            )
            loss_sums, interval_updates = {}, 0
            interval_started = time.perf_counter()

        is_validation_update = update % settings.validation_interval == 0 or update == updates
        if validate is not None and is_validation_update:
            started = time.perf_counter()
            metrics = validate(model)
            outcome.validation_seconds += time.perf_counter() - started
            outcome.validation_history.append({"update": update, **metrics})
            position = _position(update, updates, updates_per_epoch)
            _report_validation(position, metrics, not encoder_trains)
            interval_started += time.perf_counter() - started
            if outcome.best_state is None or metrics["uar"] > outcome.best_metrics["uar"]:
                outcome.best_state = {
                    name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                }
                outcome.best_update = update
                outcome.best_metrics = metrics
    for parameter in encoder_parameters:
        parameter.requires_grad_(True)
    if device.type == "cuda":
        outcome.peak_gpu_memory_gib = torch.cuda.max_memory_allocated(device) / 2**30
    return outcome
