"""Loss functions for multi-task teachers and distilled students."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as functional

from compact_ser.models import SpeechModelOutput, create_frame_mask


def transcript_loss(
    output: SpeechModelOutput,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_token: int = 0,
) -> torch.Tensor:
    """Compute the CTC loss for speech recognition.

    Args:
        output: Model outputs containing transcript logits.
        targets: Padded token targets with shape `(batch, tokens)`.
        target_lengths: Number of valid tokens in each target.
        blank_token: Index of the CTC blank token.

    Returns:
        The mean CTC loss.

    Raises:
        ValueError: If the model has no transcript head.
    """
    if output.transcript_logits is None:
        raise ValueError("The model output has no transcript logits.")
    log_probabilities = output.transcript_logits.log_softmax(dim=-1).transpose(0, 1)
    return functional.ctc_loss(
        log_probabilities,
        targets,
        output.frame_lengths,
        target_lengths,
        blank=blank_token,
        zero_infinity=True,
    )


def multitask_loss(
    output: SpeechModelOutput,
    emotion_labels: torch.Tensor,
    transcript_targets: torch.Tensor | None = None,
    transcript_lengths: torch.Tensor | None = None,
    speaker_labels: torch.Tensor | None = None,
    transcript_weight: float = 0.0,
    speaker_weight: float = 0.0,
) -> torch.Tensor:
    """Combine the emotion loss with any active auxiliary losses.

    Args:
        output: Model outputs for the batch.
        emotion_labels: Emotion class indices.
        transcript_targets: Padded token targets for speech recognition.
        transcript_lengths: Number of valid tokens in each target.
        speaker_labels: Speaker class indices.
        transcript_weight: Weight of the speech recognition loss.
        speaker_weight: Weight of the speaker identification loss.

    Returns:
        The weighted total loss.
    """
    loss = functional.cross_entropy(output.emotion_logits, emotion_labels)
    if transcript_weight > 0.0:
        loss = loss + transcript_weight * transcript_loss(
            output, transcript_targets, transcript_lengths
        )
    if speaker_weight > 0.0:
        loss = loss + speaker_weight * functional.cross_entropy(
            output.speaker_logits, speaker_labels
        )
    return loss


def hidden_state_loss(
    student_states: Sequence[torch.Tensor],
    teacher_states: Sequence[torch.Tensor],
    teacher_layers: Sequence[int],
    frame_lengths: torch.Tensor,
) -> torch.Tensor:
    """Compute the mean squared error between matched layers, ignoring padded frames.

    Args:
        student_states: Student embedding output followed by each layer output.
        teacher_states: Teacher embedding output followed by each layer output.
        teacher_layers: One-based teacher layer matched to each student layer, in order.
        frame_lengths: Number of valid frames in each utterance.

    Returns:
        The error averaged across valid frames and matched layers.
    """
    frame_count = student_states[0].size(1)
    mask = create_frame_mask(frame_lengths, frame_count).to(student_states[0].dtype)
    total = student_states[0].new_zeros(())
    for student_layer, teacher_layer in enumerate(teacher_layers, start=1):
        squared_error = (student_states[student_layer] - teacher_states[teacher_layer]).pow(2)
        total = total + (squared_error.mean(dim=-1) * mask).sum() / mask.sum().clamp_min(1.0)
    return total / len(teacher_layers)


def distillation_loss(
    student_output: SpeechModelOutput,
    teacher_output: SpeechModelOutput,
    teacher_layers: Sequence[int],
    emotion_labels: torch.Tensor,
    temperature: float,
    logit_weight: float,
    hidden_state_weight: float,
) -> torch.Tensor:
    """Combine cross-entropy, temperature-scaled KL divergence, and hidden-state error.

    Args:
        student_output: Student outputs, including hidden states.
        teacher_output: Teacher outputs, including hidden states, computed without gradients.
        teacher_layers: One-based teacher layer matched to each student layer, in order.
        emotion_labels: Emotion class indices.
        temperature: Softening temperature for the emotion logits.
        logit_weight: Weight of the KL divergence.
        hidden_state_weight: Weight of the hidden-state error.

    Returns:
        The weighted total loss.
    """
    supervised = functional.cross_entropy(student_output.emotion_logits, emotion_labels)
    soft_targets = (teacher_output.emotion_logits / temperature).softmax(dim=-1)
    soft_predictions = (student_output.emotion_logits / temperature).log_softmax(dim=-1)
    logit_term = functional.kl_div(soft_predictions, soft_targets, reduction="batchmean")
    hidden_term = hidden_state_loss(
        student_output.hidden_states,
        teacher_output.hidden_states,
        teacher_layers,
        student_output.frame_lengths,
    )
    return (
        supervised + logit_weight * temperature**2 * logit_term + hidden_state_weight * hidden_term
    )
