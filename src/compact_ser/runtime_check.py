"""Check a training runtime and measure training updates on synthetic audio.

The check uses no corpus data. It records the environment, loads the pinned checkpoint, and times
one multi-task teacher and each student depth. The results are used to estimate compute needs.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import Wav2Vec2Model

from compact_ser.losses import distillation_loss, multitask_loss
from compact_ser.models import (
    SpeechEmotionModel,
    build_student,
    load_pretrained_encoder,
    select_evenly_spaced_layers,
    small_encoder_config,
)
from compact_ser.runtime_environment import describe_runtime, save_runtime_record
from compact_ser.study_design import load_study_design

COMPLETION_MARKER = "RUNTIME CHECK COMPLETE"
FAILURE_MARKER = "RUNTIME CHECK FAILED"
RESULT_FILE_NAME = "runtime_check.json"
SAMPLE_RATE = 16_000
VOCABULARY_SIZE = 32
TRAINING_SPEAKERS = 6
TARGET_TOKENS = 40
LEARNING_RATE = 1e-5
DISTILLATION_TEMPERATURE = 2.0


@dataclass(frozen=True)
class MeasurementSettings:
    """Size and length of the timed training updates."""

    batch_size: int
    duration_seconds: float
    updates: int
    warmup_updates: int


@dataclass(frozen=True)
class SyntheticBatch:
    """A batch of random audio with variable lengths and random labels."""

    waveforms: torch.Tensor
    attention_mask: torch.Tensor
    emotion_labels: torch.Tensor
    speaker_labels: torch.Tensor
    transcript_targets: torch.Tensor
    transcript_lengths: torch.Tensor


def create_synthetic_batch(
    settings: MeasurementSettings, num_emotions: int, device: torch.device
) -> SyntheticBatch:
    """Create a random batch whose utterances fill 50% to 100% of the padded length.

    Args:
        settings: Batch size and maximum utterance length.
        num_emotions: Number of emotion classes.
        device: Device that receives the tensors.

    Returns:
        The synthetic batch.
    """
    samples = int(settings.duration_seconds * SAMPLE_RATE)
    lengths = torch.randint(samples // 2, samples + 1, (settings.batch_size,))
    lengths[0] = samples
    positions = torch.arange(samples).unsqueeze(0)
    attention_mask = (positions < lengths.unsqueeze(1)).long()
    waveforms = 0.1 * torch.randn(settings.batch_size, samples) * attention_mask
    return SyntheticBatch(
        waveforms=waveforms.to(device),
        attention_mask=attention_mask.to(device),
        emotion_labels=torch.randint(num_emotions, (settings.batch_size,), device=device),
        speaker_labels=torch.randint(TRAINING_SPEAKERS, (settings.batch_size,), device=device),
        transcript_targets=torch.randint(
            1, VOCABULARY_SIZE, (settings.batch_size, TARGET_TOKENS), device=device
        ),
        transcript_lengths=torch.full((settings.batch_size,), TARGET_TOKENS, device=device),
    )


def _time_updates(
    update: Callable[[], None], settings: MeasurementSettings, device: torch.device
) -> dict[str, float]:
    uses_gpu = device.type == "cuda"
    if uses_gpu:
        torch.cuda.reset_peak_memory_stats(device)
    durations = []
    for _ in range(settings.warmup_updates + settings.updates):
        start = time.perf_counter()
        update()
        if uses_gpu:
            torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - start)
    peak_memory = torch.cuda.max_memory_allocated(device) / 2**30 if uses_gpu else 0.0
    return {
        "seconds_per_update": statistics.median(durations[settings.warmup_updates :]),
        "peak_gpu_memory_gib": peak_memory,
    }


def measure_teacher(
    encoder: Wav2Vec2Model,
    num_emotions: int,
    settings: MeasurementSettings,
    device: torch.device,
) -> dict[str, float]:
    """Time updates of a teacher with emotion, transcript, and speaker heads.

    Args:
        encoder: Encoder used by the teacher.
        num_emotions: Number of emotion classes.
        settings: Size and length of the timed updates.
        device: Device used for training.

    Returns:
        Median seconds per update and peak GPU memory.
    """
    model = SpeechEmotionModel(encoder, num_emotions, VOCABULARY_SIZE, TRAINING_SPEAKERS).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    batch = create_synthetic_batch(settings, num_emotions, device)

    def update() -> None:
        optimizer.zero_grad(set_to_none=True)
        output = model(batch.waveforms, batch.attention_mask)
        loss = multitask_loss(
            output,
            batch.emotion_labels,
            batch.transcript_targets,
            batch.transcript_lengths,
            batch.speaker_labels,
            transcript_weight=1.0,
            speaker_weight=1.0,
        )
        loss.backward()
        optimizer.step()

    return _time_updates(update, settings, device)


def measure_student(
    encoder: Wav2Vec2Model,
    depth: int,
    num_emotions: int,
    settings: MeasurementSettings,
    device: torch.device,
) -> dict[str, float]:
    """Time distillation updates, including the forward pass through the teacher.

    Args:
        encoder: Encoder used by the teacher and copied by the student.
        depth: Number of Transformer layers in the student.
        num_emotions: Number of emotion classes.
        settings: Size and length of the timed updates.
        device: Device used for training.

    Returns:
        Median seconds per update and peak GPU memory.
    """
    teacher = SpeechEmotionModel(encoder, num_emotions).to(device).eval()
    student = build_student(encoder, depth, num_emotions).to(device).train()
    teacher_layers = select_evenly_spaced_layers(len(encoder.encoder.layers), depth)
    optimizer = torch.optim.AdamW(student.parameters(), lr=LEARNING_RATE)
    batch = create_synthetic_batch(settings, num_emotions, device)

    def update() -> None:
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher_output = teacher(
                batch.waveforms, batch.attention_mask, output_hidden_states=True
            )
        student_output = student(batch.waveforms, batch.attention_mask, output_hidden_states=True)
        loss = distillation_loss(
            student_output,
            teacher_output,
            teacher_layers,
            batch.emotion_labels,
            temperature=DISTILLATION_TEMPERATURE,
            logit_weight=1.0,
            hidden_state_weight=1.0,
        )
        loss.backward()
        optimizer.step()

    return _time_updates(update, settings, device)


def _release_gpu_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def run_check(
    output_dir: Path, settings: MeasurementSettings, use_small_model: bool = False
) -> dict[str, object]:
    """Run the full check and save its results.

    Args:
        output_dir: Folder that receives the environment record and results.
        settings: Size and length of the timed updates.
        use_small_model: Whether to use a small random model instead of the pinned checkpoint.

    Returns:
        The saved results.
    """
    design = load_study_design()
    output_dir.mkdir(parents=True, exist_ok=True)
    record_dir = save_runtime_record(output_dir)
    runtime = describe_runtime()
    print(f"Runtime: {json.dumps(runtime, sort_keys=True)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(design.tuning_seeds[0])
    start = time.perf_counter()
    if use_small_model:
        encoder = Wav2Vec2Model(small_encoder_config())
    else:
        encoder = load_pretrained_encoder(design.checkpoint_name, design.checkpoint_revision)
    print(f"Encoder ready in {time.perf_counter() - start:.1f} s.", flush=True)

    num_emotions = len(design.emotions)
    measurements: dict[str, dict[str, float]] = {}
    measurements["teacher_ser_asr_sid"] = measure_teacher(encoder, num_emotions, settings, device)
    _release_gpu_memory(device)
    print(f"Teacher: {measurements['teacher_ser_asr_sid']}", flush=True)
    encoder.cpu()
    for depth in design.depths:
        name = f"student_depth{depth}"
        measurements[name] = measure_student(encoder, depth, num_emotions, settings, device)
        encoder.cpu()
        _release_gpu_memory(device)
        print(f"Student with {depth} layers: {measurements[name]}", flush=True)

    results = {
        "checkpoint": "small random model" if use_small_model else design.checkpoint_name,
        "checkpoint_revision": None if use_small_model else design.checkpoint_revision,
        "device": device.type,
        "environment_record": record_dir.name,
        "runtime": runtime,
        "settings": asdict(settings),
        "measurements": measurements,
    }
    result_path = output_dir / RESULT_FILE_NAME
    result_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results


def main(arguments: Sequence[str] | None = None) -> None:
    """Parse command-line arguments and run the check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--duration-seconds", type=float, default=6.0)
    parser.add_argument("--updates", type=int, default=30)
    parser.add_argument("--warmup-updates", type=int, default=5)
    parser.add_argument(
        "--small-model",
        action="store_true",
        help="Use a small random model instead of downloading the checkpoint. For tests only.",
    )
    options = parser.parse_args(arguments)
    settings = MeasurementSettings(
        batch_size=options.batch_size,
        duration_seconds=options.duration_seconds,
        updates=options.updates,
        warmup_updates=options.warmup_updates,
    )
    try:
        run_check(options.output_dir, settings, use_small_model=options.small_model)
    except Exception:
        traceback.print_exc()
        print(FAILURE_MARKER, flush=True)
        raise
    print(COMPLETION_MARKER, flush=True)


if __name__ == "__main__":
    main()
