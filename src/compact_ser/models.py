"""Wav2vec 2.0 speech emotion models with optional auxiliary task heads."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
from transformers import Wav2Vec2Config, Wav2Vec2Model


@dataclass
class SpeechModelOutput:
    """Outputs of a speech emotion model.

    Attributes:
        emotion_logits: Emotion scores with shape `(batch, emotions)`.
        frame_lengths: Number of valid encoder frames in each utterance.
        transcript_logits: Frame-level token scores, when the model has a transcript head.
        speaker_logits: Speaker scores, when the model has a speaker head.
        hidden_states: Embedding output followed by each Transformer layer output, when requested.
    """

    emotion_logits: torch.Tensor
    frame_lengths: torch.Tensor
    transcript_logits: torch.Tensor | None = None
    speaker_logits: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None


def compute_frame_lengths(sample_lengths: torch.Tensor, config: Wav2Vec2Config) -> torch.Tensor:
    """Convert waveform lengths to encoder frame counts.

    Args:
        sample_lengths: Number of valid audio samples in each utterance.
        config: Encoder configuration describing the convolutional frontend.

    Returns:
        Number of frames produced for each utterance.
    """
    lengths = sample_lengths
    for kernel, stride in zip(config.conv_kernel, config.conv_stride, strict=True):
        lengths = torch.div(lengths - kernel, stride, rounding_mode="floor") + 1
    return lengths


def create_frame_mask(frame_lengths: torch.Tensor, frame_count: int) -> torch.Tensor:
    """Create a boolean mask that marks valid frames.

    Args:
        frame_lengths: Number of valid frames in each utterance.
        frame_count: Number of frames in the padded batch.

    Returns:
        A boolean tensor with shape `(batch, frames)`.
    """
    positions = torch.arange(frame_count, device=frame_lengths.device)
    return positions.unsqueeze(0) < frame_lengths.unsqueeze(1)


def select_evenly_spaced_layers(total_layers: int, depth: int) -> tuple[int, ...]:
    """Choose evenly spaced Transformer layers, ending with the last layer.

    Args:
        total_layers: Number of layers in the source encoder.
        depth: Number of layers to keep.

    Returns:
        One-based layer numbers. For example, 12 layers and a depth of 4 give `(3, 6, 9, 12)`.

    Raises:
        ValueError: If the layers cannot be spaced evenly.
    """
    if depth < 1 or total_layers % depth != 0:
        raise ValueError(f"Cannot space {depth} layers evenly across {total_layers} layers.")
    step = total_layers // depth
    return tuple(range(step, total_layers + 1, step))


class SpeechEmotionModel(nn.Module):
    """Wav2vec 2.0 encoder with an emotion head and optional auxiliary heads."""

    def __init__(
        self,
        encoder: Wav2Vec2Model,
        num_emotions: int,
        vocabulary_size: int | None = None,
        num_speakers: int | None = None,
    ) -> None:
        super().__init__()
        hidden_size = encoder.config.hidden_size
        self.encoder = encoder
        self.emotion_head = nn.Linear(hidden_size, num_emotions)
        self.transcript_head = nn.Linear(hidden_size, vocabulary_size) if vocabulary_size else None
        self.speaker_head = nn.Linear(hidden_size, num_speakers) if num_speakers else None

    def forward(
        self,
        waveforms: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> SpeechModelOutput:
        """Run the encoder and every available head.

        The pinned checkpoint normalizes frontend features with group normalization, which also
        sees padded samples. Padding therefore changes its predictions, so evaluation must process
        each utterance without padding.

        Args:
            waveforms: Padded audio with shape `(batch, samples)`.
            attention_mask: Mask with ones for valid samples and zeros for padding.
            output_hidden_states: Whether to return the output of each layer.

        Returns:
            The model outputs for the batch.
        """
        encoded = self.encoder(
            input_values=waveforms,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )
        frames = encoded.last_hidden_state
        frame_lengths = compute_frame_lengths(
            attention_mask.sum(dim=-1).long(), self.encoder.config
        )
        mask = create_frame_mask(frame_lengths, frames.size(1)).unsqueeze(-1).to(frames.dtype)
        pooled = (frames * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return SpeechModelOutput(
            emotion_logits=self.emotion_head(pooled),
            frame_lengths=frame_lengths,
            transcript_logits=self.transcript_head(frames) if self.transcript_head else None,
            speaker_logits=self.speaker_head(pooled) if self.speaker_head else None,
            hidden_states=encoded.hidden_states,
        )


def load_pretrained_encoder(name: str, revision: str) -> Wav2Vec2Model:
    """Load a pinned wav2vec 2.0 checkpoint.

    Args:
        name: Checkpoint name on the Hugging Face Hub.
        revision: Exact commit of the checkpoint.

    Returns:
        The pretrained encoder.
    """
    # Scaled dot-product attention does not store attention matrices, which saves memory on long
    # utterances.
    return Wav2Vec2Model.from_pretrained(name, revision=revision, attn_implementation="sdpa")


def build_student(
    source_encoder: Wav2Vec2Model, depth: int, num_emotions: int
) -> SpeechEmotionModel:
    """Create an emotion-only student by copying evenly spaced encoder layers.

    The source encoder is not changed. The frontend and selected layers keep their weights, and the
    emotion head is newly initialized.

    Args:
        source_encoder: Encoder to copy from.
        depth: Number of Transformer layers in the student.
        num_emotions: Number of emotion classes.

    Returns:
        The student model.
    """
    encoder = copy.deepcopy(source_encoder)
    layer_numbers = select_evenly_spaced_layers(len(encoder.encoder.layers), depth)
    encoder.encoder.layers = nn.ModuleList(
        encoder.encoder.layers[number - 1] for number in layer_numbers
    )
    encoder.config.num_hidden_layers = depth
    return SpeechEmotionModel(encoder, num_emotions)


def small_encoder_config(feature_norm: str = "group") -> Wav2Vec2Config:
    """Return a small 12-layer configuration for tests that need no downloaded weights.

    Like the pinned checkpoint, it uses group normalization and no layer dropout by default.

    Args:
        feature_norm: Normalization used by the convolutional frontend.

    Returns:
        The encoder configuration.
    """
    return Wav2Vec2Config(
        feat_extract_norm=feature_norm,
        layerdrop=0.0,
        hidden_size=32,
        num_hidden_layers=12,
        num_attention_heads=2,
        intermediate_size=64,
        conv_dim=(16, 16, 16, 16, 16, 16, 16),
    )
