"""Execute the planned training runs and resume after an interrupted session.

    python -m compact_ser.study_runner --manifest data/manifests/iemocap.jsonl \
        --corpus-root data/iemocap --output-root outputs --scope core

Each run writes to its own folder, and the ledger records its progress. An interrupted run restarts
from the beginning with the same seed. Held-out test sessions are never read by this module.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from transformers import Wav2Vec2Model

from compact_ser.audio_data import (
    EncodedUtterance,
    SpeechBatch,
    compute_class_weights,
    encode_utterances,
    iterate_batches,
)
from compact_ser.colab_jobs import CODE_COMMIT_VARIABLE
from compact_ser.corpora import Partition, Utterance, read_manifest, select_partition
from compact_ser.losses import distillation_loss_terms, multitask_loss_terms, transcript_loss
from compact_ser.models import (
    SpeechEmotionModel,
    build_student,
    load_pretrained_encoder,
    select_evenly_spaced_layers,
    small_encoder_config,
)
from compact_ser.run_ledger import RunLedger, RunStatus
from compact_ser.study_design import (
    DistillationCandidate,
    PlanScope,
    RunKind,
    RunSpec,
    SessionFold,
    StudyDesign,
    TeacherObjective,
    build_run_plan,
    load_study_design,
    session_folds,
)
from compact_ser.text_processing import TranscriptTokenizer
from compact_ser.training import TrainingOutcome, evaluate_model, seed_everything, train_stage

COMPLETION_MARKER = "STUDY RUNS COMPLETE"
FAILURE_MARKER = "STUDY RUNS FAILED"
RESULT_FILE_NAME = "result.json"
MODEL_FILE_NAME = "model.pt"
LEDGER_FILE_NAME = "run_ledger.jsonl"
SOURCE_ENCODER_DEPTH = 12
USES_TRANSCRIPTS = {
    TeacherObjective.SER_ASR_SID,
    TeacherObjective.SER_ASR,
    TeacherObjective.ASR_THEN_SER,
}
USES_SPEAKERS = {TeacherObjective.SER_ASR_SID, TeacherObjective.SER_SID}


@dataclass
class FoldData:
    """Training and validation data for one fold."""

    tokenizer: TranscriptTokenizer
    speakers: dict[str, int]
    training: list[EncodedUtterance]
    validation: list[EncodedUtterance]
    class_weights: torch.Tensor


@dataclass
class StudyContext:
    """Everything needed to execute planned runs."""

    design: StudyDesign
    manifest: list[Utterance]
    corpus_root: Path
    output_root: Path
    device: torch.device
    encoder_factory: Callable[[], Wav2Vec2Model]
    _cached_fold: tuple[int, FoldData] | None = field(default=None, repr=False)

    @property
    def num_emotions(self) -> int:
        """Number of emotion classes."""
        return len(self.design.emotions)

    def run_folder(self, run_id: str) -> Path:
        """Return the output folder of a run."""
        return self.output_root / "runs" / run_id

    def fold_data(self, test_session: int) -> FoldData:
        """Load the training and validation data of a fold, reusing the latest fold loaded.

        The tokenizer, speaker classes, and class weights come from the training sessions only.

        Args:
            test_session: The session held out by the fold.

        Returns:
            The prepared data.
        """
        if self._cached_fold is not None and self._cached_fold[0] == test_session:
            return self._cached_fold[1]
        fold = _fold_for_session(self.design, test_session)
        training = select_partition(self.manifest, fold, Partition.TRAINING)
        validation = select_partition(self.manifest, fold, Partition.VALIDATION)
        tokenizer = TranscriptTokenizer.fit(utterance.transcript for utterance in training)
        speakers = {
            speaker: index
            for index, speaker in enumerate(sorted({utterance.speaker for utterance in training}))
        }
        emotions = self.design.emotions
        encoded_training = encode_utterances(
            training, self.corpus_root, emotions, tokenizer, speakers
        )
        data = FoldData(
            tokenizer=tokenizer,
            speakers=speakers,
            training=encoded_training,
            validation=encode_utterances(validation, self.corpus_root, emotions, tokenizer),
            class_weights=compute_class_weights(encoded_training, len(emotions)).to(self.device),
        )
        self._cached_fold = (test_session, data)
        return data


def _fold_for_session(design: StudyDesign, test_session: int) -> SessionFold:
    return next(
        fold for fold in session_folds(design.sessions) if fold.test_session == test_session
    )


def _load_result(context: StudyContext, run_id: str) -> dict[str, Any]:
    path = context.run_folder(run_id) / RESULT_FILE_NAME
    if not path.exists():
        raise FileNotFoundError(f"{run_id} must complete before the runs that depend on it.")
    return json.loads(path.read_text(encoding="utf-8"))


def _select_candidate(
    context: StudyContext, candidates: Sequence[RunSpec]
) -> tuple[RunSpec, dict[str, Any]]:
    """Return the candidate with the best validation UAR, preferring the first on a tie."""
    results = [_load_result(context, run.run_id) for run in candidates]
    best = max(range(len(candidates)), key=lambda i: results[i]["validation_metrics"]["uar"])
    return candidates[best], results[best]


def _teacher_tuning_candidates(context: StudyContext, run: RunSpec) -> list[RunSpec]:
    return [
        candidate
        for candidate in build_run_plan(context.design)
        if candidate.kind is RunKind.TEACHER_TUNING
        and candidate.objective is run.objective
        and candidate.test_session == run.test_session
    ]


def _distillation_tuning_candidates(context: StudyContext, run: RunSpec) -> list[RunSpec]:
    return [
        candidate
        for candidate in build_run_plan(context.design)
        if candidate.kind is RunKind.DISTILLATION_TUNING
        and candidate.objective is run.objective
        and candidate.test_session == run.test_session
    ]


def _calibrate_auxiliary_scales(
    model: SpeechEmotionModel,
    batches: Sequence[SpeechBatch],
    class_weights: torch.Tensor,
) -> dict[str, float]:
    """Find the factors that make each auxiliary loss equal the initial emotion loss."""
    model.eval()
    emotion_total = transcript_total = speaker_total = 0.0
    with torch.no_grad():
        for batch in batches:
            output = model(batch.waveforms, batch.attention_mask)
            emotion_total += float(
                functional.cross_entropy(
                    output.emotion_logits, batch.emotion_labels, weight=class_weights
                )
            )
            if model.transcript_head is not None:
                transcript_total += float(
                    transcript_loss(output, batch.transcript_targets, batch.transcript_lengths)
                )
            if model.speaker_head is not None:
                speaker_total += float(
                    functional.cross_entropy(output.speaker_logits, batch.speaker_labels)
                )
    model.train()
    return {
        "transcript": emotion_total / transcript_total if transcript_total else 0.0,
        "speaker": emotion_total / speaker_total if speaker_total else 0.0,
    }


def _save_run(
    context: StudyContext,
    run: RunSpec,
    outcome: TrainingOutcome,
    details: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    folder = context.run_folder(run.run_id)
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(outcome.best_state, folder / MODEL_FILE_NAME)
    result = {
        "run_id": run.run_id,
        "validation_metrics": outcome.best_metrics,
        "best_update": outcome.best_update,
        "validation_history": outcome.validation_history,
        "wall_seconds": time.perf_counter() - started,
        "code_commit": os.environ.get(CODE_COMMIT_VARIABLE),
        **details,
    }
    # The result is written last, so its presence shows that the model file is complete.
    (folder / RESULT_FILE_NAME).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _train_teacher(context: StudyContext, run: RunSpec) -> dict[str, Any]:
    started = time.perf_counter()
    design = context.design
    objective = run.objective
    candidate = run.candidate
    if run.kind is RunKind.TEACHER:
        selected, _ = _select_candidate(context, _teacher_tuning_candidates(context, run))
        candidate = selected.candidate
    data = context.fold_data(run.test_session)
    settings = design.training

    seed_everything(run.seed)
    uses_transcripts = objective in USES_TRANSCRIPTS
    uses_speakers = objective in USES_SPEAKERS
    vocabulary_size = data.tokenizer.vocabulary_size if uses_transcripts else None
    num_speakers = len(data.speakers) if uses_speakers else None
    encoder = context.encoder_factory()
    encoder.freeze_feature_encoder()
    model = SpeechEmotionModel(encoder, context.num_emotions, vocabulary_size, num_speakers)
    model.to(context.device)

    def validate(current_model: SpeechEmotionModel) -> dict[str, float]:
        tokenizer = data.tokenizer if uses_transcripts else None
        return evaluate_model(
            current_model, data.validation, context.num_emotions, context.device, tokenizer
        )["metrics"]

    batches = iterate_batches(data.training, settings.batch_size, run.seed, context.device)
    details: dict[str, Any] = {
        "architecture": {"vocabulary_size": vocabulary_size, "num_speakers": num_speakers},
        "candidate": candidate,
    }

    if objective is TeacherObjective.ASR_THEN_SER:
        share = design.asr_update_shares[candidate]
        adaptation_updates = round(share * settings.teacher_updates)
        details["asr_update_share"] = share

        def adaptation_loss(batch: SpeechBatch) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
            output = model(batch.waveforms, batch.attention_mask)
            loss = transcript_loss(output, batch.transcript_targets, batch.transcript_lengths)
            return {"transcript": loss}, {}

        adaptation = train_stage(model, adaptation_loss, batches, adaptation_updates, settings)
        details["adaptation_gradient_history"] = adaptation.gradient_history
        remaining_updates = settings.teacher_updates - adaptation_updates
        transcript_weight = speaker_weight = 0.0
    else:
        remaining_updates = settings.teacher_updates
        transcript_weight = speaker_weight = 0.0
        if uses_transcripts or uses_speakers:
            calibration_stream = iterate_batches(
                data.training, settings.batch_size, run.seed, context.device
            )
            calibration_batches = [
                next(calibration_stream) for _ in range(settings.calibration_batches)
            ]
            scales = _calibrate_auxiliary_scales(model, calibration_batches, data.class_weights)
            auxiliary_scale = design.auxiliary_loss_scales[candidate]
            transcript_weight = auxiliary_scale * scales["transcript"]
            speaker_weight = auxiliary_scale * scales["speaker"]
            details["auxiliary_loss_scale"] = auxiliary_scale
            details["transcript_weight"] = transcript_weight
            details["speaker_weight"] = speaker_weight

    def compute_loss(batch: SpeechBatch) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        output = model(batch.waveforms, batch.attention_mask)
        terms = multitask_loss_terms(
            output,
            batch.emotion_labels,
            batch.transcript_targets,
            batch.transcript_lengths,
            batch.speaker_labels,
            transcript_weight=transcript_weight,
            speaker_weight=speaker_weight,
            class_weights=data.class_weights,
        )
        diagnostics = {}
        if speaker_weight > 0.0:
            predictions = output.speaker_logits.argmax(dim=-1)
            accuracy = (predictions == batch.speaker_labels).float().mean()
            diagnostics["speaker_training_accuracy"] = float(accuracy)
        return terms, diagnostics

    outcome = train_stage(model, compute_loss, batches, remaining_updates, settings, validate)
    details["gradient_history"] = outcome.gradient_history
    return _save_run(context, run, outcome, details, started)


def _load_teacher(context: StudyContext, run_id: str) -> SpeechEmotionModel:
    architecture = _load_result(context, run_id)["architecture"]
    teacher = SpeechEmotionModel(
        context.encoder_factory(),
        context.num_emotions,
        architecture["vocabulary_size"],
        architecture["num_speakers"],
    )
    state = torch.load(
        context.run_folder(run_id) / MODEL_FILE_NAME, map_location="cpu", weights_only=True
    )
    teacher.load_state_dict(state)
    return teacher


def _source_teacher_id(context: StudyContext, run: RunSpec) -> str | None:
    """Return the teacher that provides a student's weights and distillation targets."""
    if run.kind in (RunKind.PILOT_STUDENT, RunKind.DISTILLATION_TUNING):
        selected, _ = _select_candidate(context, _teacher_tuning_candidates(context, run))
        return selected.run_id
    if run.kind in (RunKind.DISTILLED_STUDENT, RunKind.INHERITED_STUDENT):
        return RunSpec(
            RunKind.TEACHER, run.test_session, objective=run.objective, seed=run.seed
        ).run_id
    return None


def _distillation_weights(context: StudyContext, run: RunSpec) -> DistillationCandidate | None:
    candidates = context.design.distillation_candidates
    if run.kind is RunKind.PILOT_STUDENT:
        return candidates[0]
    if run.kind is RunKind.DISTILLATION_TUNING:
        return candidates[run.candidate]
    if run.kind is RunKind.DISTILLED_STUDENT:
        selected, _ = _select_candidate(context, _distillation_tuning_candidates(context, run))
        return candidates[selected.candidate]
    return None


def _train_student(context: StudyContext, run: RunSpec) -> dict[str, Any]:
    started = time.perf_counter()
    settings = context.design.training
    data = context.fold_data(run.test_session)
    teacher_id = _source_teacher_id(context, run)
    weights = _distillation_weights(context, run)

    seed_everything(run.seed)
    teacher = _load_teacher(context, teacher_id) if teacher_id else None
    source_encoder = teacher.encoder if teacher else context.encoder_factory()
    student = build_student(source_encoder, run.depth, context.num_emotions)
    student.encoder.freeze_feature_encoder()
    student.to(context.device)
    teacher_layers = select_evenly_spaced_layers(SOURCE_ENCODER_DEPTH, run.depth)
    distils = weights is not None
    if distils:
        teacher.requires_grad_(False).eval().to(context.device)

    def compute_loss(batch: SpeechBatch) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        output = student(batch.waveforms, batch.attention_mask, output_hidden_states=distils)
        if not distils:
            return multitask_loss_terms(
                output, batch.emotion_labels, class_weights=data.class_weights
            ), {}
        with torch.no_grad():
            teacher_output = teacher(
                batch.waveforms, batch.attention_mask, output_hidden_states=True
            )
        terms = distillation_loss_terms(
            output,
            teacher_output,
            teacher_layers,
            batch.emotion_labels,
            temperature=context.design.distillation_temperature,
            logit_weight=weights.logit_weight,
            hidden_state_weight=weights.hidden_state_weight,
            class_weights=data.class_weights,
        )
        return terms, {}

    def validate(current_model: SpeechEmotionModel) -> dict[str, float]:
        return evaluate_model(current_model, data.validation, context.num_emotions, context.device)[
            "metrics"
        ]

    batches = iterate_batches(data.training, settings.batch_size, run.seed, context.device)
    outcome = train_stage(
        student, compute_loss, batches, settings.student_updates, settings, validate
    )
    details = {
        "source_teacher": teacher_id,
        "depth": run.depth,
        "candidate": run.candidate,
        "distillation_weights": vars(weights) if weights else None,
        "gradient_history": outcome.gradient_history,
    }
    return _save_run(context, run, outcome, details, started)


def execute_run(context: StudyContext, run: RunSpec) -> dict[str, Any]:
    """Train the model for one planned run and save it.

    Args:
        context: The study context.
        run: The run to execute.

    Returns:
        The saved result.
    """
    if run.kind in (RunKind.TEACHER_TUNING, RunKind.TEACHER):
        return _train_teacher(context, run)
    return _train_student(context, run)


def execute_pending_runs(
    context: StudyContext, plan: Sequence[RunSpec], max_runs: int | None = None
) -> int:
    """Execute planned runs in order, skipping runs that have already completed.

    Args:
        context: The study context.
        plan: Run specifications in execution order.
        max_runs: Largest number of runs to execute in this call.

    Returns:
        The number of runs executed.
    """
    ledger = RunLedger(context.output_root / LEDGER_FILE_NAME)
    pending = ledger.pending_runs(plan)
    if max_runs is not None:
        pending = pending[:max_runs]
    for position, run in enumerate(pending, start=1):
        ledger.record(run.run_id, RunStatus.STARTED)
        try:
            result = execute_run(context, run)
        except Exception as error:
            ledger.record(run.run_id, RunStatus.FAILED, error=repr(error))
            raise
        ledger.record(run.run_id, RunStatus.COMPLETED, wall_seconds=result["wall_seconds"])
        print(
            f"Run {position}/{len(pending)} {run.run_id}: "
            f"validation UAR {result['validation_metrics']['uar']:.4f}, "
            f"{result['wall_seconds']:.0f} s",
            flush=True,
        )
    return len(pending)


def _small_encoder() -> Wav2Vec2Model:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return Wav2Vec2Model(small_encoder_config())


def build_context(
    manifest: list[Utterance],
    corpus_root: Path,
    output_root: Path,
    update_limit: int | None = None,
    small_model: bool = False,
) -> StudyContext:
    """Create the study context from the study settings, using a GPU when one is available.

    Args:
        manifest: Utterances of the development corpus.
        corpus_root: Folder that the manifest's audio paths are relative to.
        output_root: Folder that receives run folders and the run ledger.
        update_limit: If given, limit every run to this many updates. For checks only.
        small_model: Use a small random model instead of the pinned checkpoint. For checks only.

    Returns:
        The study context.
    """
    design = load_study_design()
    if update_limit is not None:
        design = replace(
            design,
            training=replace(
                design.training, teacher_updates=update_limit, student_updates=update_limit
            ),
        )
    if small_model:
        encoder_factory = _small_encoder
    else:

        def encoder_factory() -> Wav2Vec2Model:
            return load_pretrained_encoder(design.checkpoint_name, design.checkpoint_revision)

    return StudyContext(
        design=design,
        manifest=manifest,
        corpus_root=corpus_root,
        output_root=output_root,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        encoder_factory=encoder_factory,
    )


def add_check_options(parser: argparse.ArgumentParser) -> None:
    """Add the options that shorten runs for checks."""
    parser.add_argument(
        "--update-limit",
        type=int,
        help="Limit every run to this many updates. For checks only.",
    )
    parser.add_argument(
        "--small-model",
        action="store_true",
        help="Use a small random model instead of downloading the checkpoint. For checks only.",
    )


def main(arguments: Sequence[str] | None = None) -> None:
    """Parse command-line arguments and execute pending runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scope", choices=[scope.value for scope in PlanScope], default="core")
    parser.add_argument("--max-runs", type=int)
    add_check_options(parser)
    options = parser.parse_args(arguments)

    context = build_context(
        read_manifest(options.manifest),
        options.corpus_root,
        options.output_root,
        options.update_limit,
        options.small_model,
    )
    plan = build_run_plan(context.design, PlanScope(options.scope))
    try:
        executed = execute_pending_runs(context, plan, options.max_runs)
    except Exception:
        traceback.print_exc()
        print(FAILURE_MARKER, flush=True)
        raise
    print(f"{executed} runs executed.", flush=True)
    print(COMPLETION_MARKER, flush=True)


if __name__ == "__main__":
    main()
