"""Audio loading, normalization, and batching."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import soundfile
import torch
from scipy.signal import resample_poly

from compact_ser.corpora import Utterance
from compact_ser.text_processing import TranscriptTokenizer, normalize_transcript

SAMPLE_RATE = 16_000
UNKNOWN_SPEAKER = -1
# Batches per length-sorted group, as in the Hugging Face length-grouped sampler.
LENGTH_GROUP_BATCHES = 50


@dataclass(frozen=True)
class EncodedUtterance:
    """An utterance prepared for a model.

    Attributes:
        waveform: Normalized mono audio at 16 kHz.
        emotion: Emotion class index.
        speaker: Training-speaker index, or `UNKNOWN_SPEAKER` for other speakers.
        tokens: Transcript tokens, when the run has a tokenizer.
        text: Normalized transcript.
    """

    waveform: np.ndarray
    emotion: int
    speaker: int
    tokens: tuple[int, ...]
    text: str


@dataclass(frozen=True)
class SpeechBatch:
    """A padded batch of utterances."""

    waveforms: torch.Tensor
    attention_mask: torch.Tensor
    emotion_labels: torch.Tensor
    speaker_labels: torch.Tensor
    transcript_targets: torch.Tensor
    transcript_lengths: torch.Tensor


def read_waveform(path: Path) -> np.ndarray:
    """Read an audio file as 16 kHz mono floating-point samples.

    Args:
        path: Audio file.

    Returns:
        The audio samples.
    """
    samples, sample_rate = soundfile.read(path, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    if sample_rate == SAMPLE_RATE:
        return mono
    divisor = gcd(SAMPLE_RATE, sample_rate)
    resampled = resample_poly(mono, SAMPLE_RATE // divisor, sample_rate // divisor)
    return resampled.astype(np.float32)


def normalize_waveform(waveform: np.ndarray) -> np.ndarray:
    """Scale an utterance to zero mean and unit variance, as the pretrained checkpoint expects.

    Args:
        waveform: Audio samples.

    Returns:
        The normalized samples.
    """
    return (waveform - waveform.mean()) / np.sqrt(waveform.var() + 1e-7)


def encode_utterances(
    utterances: Sequence[Utterance],
    corpus_root: Path,
    emotions: Sequence[str],
    tokenizer: TranscriptTokenizer | None = None,
    speakers: Mapping[str, int] | None = None,
) -> list[EncodedUtterance]:
    """Read and prepare utterances for training or evaluation.

    Args:
        utterances: Utterances to read.
        corpus_root: Folder that the manifest's audio paths are relative to.
        emotions: Emotion classes, in index order.
        tokenizer: Tokenizer for transcript targets.
        speakers: Mapping from training speakers to class indices.

    Returns:
        The prepared utterances, in the same order.
    """
    emotion_indices = {emotion: index for index, emotion in enumerate(emotions)}
    speaker_indices = speakers or {}
    return [
        EncodedUtterance(
            waveform=normalize_waveform(read_waveform(corpus_root / utterance.audio_path)),
            emotion=emotion_indices[utterance.emotion],
            speaker=speaker_indices.get(utterance.speaker, UNKNOWN_SPEAKER),
            tokens=tuple(tokenizer.encode(utterance.transcript)) if tokenizer else (),
            text=normalize_transcript(utterance.transcript),
        )
        for utterance in utterances
    ]


def collate_batch(items: Sequence[EncodedUtterance], device: torch.device) -> SpeechBatch:
    """Pad utterances into a batch.

    Args:
        items: Utterances in the batch.
        device: Device that receives the tensors.

    Returns:
        The padded batch.
    """
    longest_audio = max(item.waveform.size for item in items)
    longest_tokens = max(max((len(item.tokens) for item in items), default=0), 1)
    waveforms = torch.zeros(len(items), longest_audio)
    attention_mask = torch.zeros(len(items), longest_audio, dtype=torch.long)
    targets = torch.zeros(len(items), longest_tokens, dtype=torch.long)
    for row, item in enumerate(items):
        samples = item.waveform.size
        waveforms[row, :samples] = torch.from_numpy(item.waveform)
        attention_mask[row, :samples] = 1
        if item.tokens:
            targets[row, : len(item.tokens)] = torch.tensor(item.tokens)
    return SpeechBatch(
        waveforms=waveforms.to(device),
        attention_mask=attention_mask.to(device),
        emotion_labels=torch.tensor([item.emotion for item in items], device=device),
        speaker_labels=torch.tensor([item.speaker for item in items], device=device),
        transcript_targets=targets.to(device),
        transcript_lengths=torch.tensor([len(item.tokens) for item in items], device=device),
    )


def _length_grouped_batches(
    lengths: Sequence[int], order: list[int], size: int, generator: torch.Generator
) -> list[list[int]]:
    """Group a shuffled order into batches of similar length, in random batch order."""
    group_size = size * LENGTH_GROUP_BATCHES
    batches = []
    for start in range(0, len(order), group_size):
        group = sorted(order[start : start + group_size], key=lengths.__getitem__, reverse=True)
        batches += [group[i : i + size] for i in range(0, len(group) - size + 1, size)]
    permutation = torch.randperm(len(batches), generator=generator).tolist()
    return [batches[index] for index in permutation]


def iterate_batches(
    items: Sequence[EncodedUtterance],
    batch_size: int,
    seed: int,
    device: torch.device,
    group_by_length: bool = False,
) -> Iterator[SpeechBatch]:
    """Yield shuffled batches indefinitely, reshuffling after each pass through the data.

    Incomplete final batches are dropped, so every update uses the same number of utterances.
    Grouping by length reduces padding: the shuffled order is cut into groups of
    `LENGTH_GROUP_BATCHES` batches, each group is sorted by length and split into batches, and the
    batches are shuffled. The batch holding the longest utterance comes first, so a batch that does
    not fit in memory fails at once.

    Args:
        items: Utterances to sample.
        batch_size: Number of utterances in a batch.
        seed: Seed for the shuffling order.
        device: Device that receives the tensors.
        group_by_length: Whether to put utterances of similar length in the same batch.

    Yields:
        One batch for each update.
    """
    generator = torch.Generator().manual_seed(seed)
    size = min(batch_size, len(items))
    lengths = [item.waveform.size for item in items]
    first_pass = True
    while True:
        order = torch.randperm(len(items), generator=generator).tolist()
        if group_by_length:
            batches = _length_grouped_batches(lengths, order, size, generator)
            if first_pass:
                longest = max(range(len(batches)), key=lambda i: lengths[batches[i][0]])
                batches.insert(0, batches.pop(longest))
        else:
            batches = [order[i : i + size] for i in range(0, len(order) - size + 1, size)]
        first_pass = False
        for batch in batches:
            yield collate_batch([items[index] for index in batch], device)


def compute_class_weights(items: Sequence[EncodedUtterance], num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency emotion class weights from training data.

    Args:
        items: Training utterances.
        num_classes: Number of emotion classes.

    Returns:
        One weight per class. Weights average to one on the training data.
    """
    counts = torch.bincount(
        torch.tensor([item.emotion for item in items]), minlength=num_classes
    ).float()
    return len(items) / (num_classes * counts.clamp_min(1.0))
