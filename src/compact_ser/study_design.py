"""Study settings, cross-validation folds, and the ordered plan of training runs."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "configs" / "study.toml"


class TeacherObjective(StrEnum):
    """Training objectives compared across teachers."""

    SER_ONLY = "ser_only"
    SER_ASR_SID = "ser_asr_sid"
    SER_ASR = "ser_asr"
    SER_SID = "ser_sid"
    ASR_THEN_SER = "asr_then_ser"


class RunKind(StrEnum):
    """Types of training run in the study."""

    TEACHER_TUNING = "teacher_tuning"
    PILOT_STUDENT = "pilot_student"
    TEACHER = "teacher"
    DISTILLATION_TUNING = "distillation_tuning"
    DISTILLED_STUDENT = "distilled_student"
    INHERITED_STUDENT = "inherited_student"
    BASE_STUDENT = "base_student"


class PlanScope(StrEnum):
    """Parts of the study that a run plan can cover.

    `PILOT` is the validation pilot: teacher tuning and pilot students for the core objectives.
    `CORE` adds the runs of the primary comparison, and `FULL` adds every other run.
    """

    PILOT = "pilot"
    CORE = "core"
    FULL = "full"


CORE_OBJECTIVES = (TeacherObjective.SER_ONLY, TeacherObjective.SER_ASR_SID)
# Listed in priority order. If compute runs short, runs are removed from the end of this tuple.
ADDITIONAL_OBJECTIVES = (
    TeacherObjective.ASR_THEN_SER,
    TeacherObjective.SER_ASR,
    TeacherObjective.SER_SID,
)


@dataclass(frozen=True)
class SessionFold:
    """Assignment of corpus sessions to one cross-validation fold."""

    test_session: int
    validation_session: int
    training_sessions: tuple[int, ...]


@dataclass(frozen=True)
class TrainingSettings:
    """Optimization settings shared by every training run."""

    batch_size: int
    learning_rate: float
    teacher_updates: int
    student_updates: int
    validation_interval: int
    gradient_log_interval: int
    max_gradient_norm: float
    calibration_batches: int
    head_only_share: float
    warmup_share: float
    hold_share: float


@dataclass(frozen=True)
class DistillationCandidate:
    """Loss weights compared when tuning distillation."""

    logit_weight: float
    hidden_state_weight: float


@dataclass(frozen=True)
class StudyDesign:
    """Fixed study settings loaded from the configuration file."""

    checkpoint_name: str
    checkpoint_revision: str
    sessions: tuple[int, ...]
    emotions: tuple[str, ...]
    confirmatory_seeds: tuple[int, ...]
    tuning_seeds: tuple[int, ...]
    auxiliary_loss_scales: tuple[float, ...]
    asr_update_shares: tuple[float, ...]
    primary_depth: int
    depths: tuple[int, ...]
    training: TrainingSettings
    distillation_temperature: float
    distillation_candidates: tuple[DistillationCandidate, ...]

    @property
    def secondary_depths(self) -> tuple[int, ...]:
        """Student depths other than the primary depth."""
        return tuple(depth for depth in self.depths if depth != self.primary_depth)


@dataclass(frozen=True)
class RunSpec:
    """One training run in the study plan."""

    kind: RunKind
    test_session: int
    objective: TeacherObjective | None = None
    depth: int | None = None
    candidate: int | None = None
    seed: int | None = None

    @property
    def run_id(self) -> str:
        """Unique identifier that is also safe to use as a folder name."""
        parts = [self.kind.value]
        if self.objective is not None:
            parts.append(self.objective.value)
        parts.append(f"session{self.test_session}")
        if self.depth is not None:
            parts.append(f"depth{self.depth}")
        if self.candidate is not None:
            parts.append(f"candidate{self.candidate}")
        if self.seed is not None:
            parts.append(f"seed{self.seed}")
        return "__".join(parts)


def load_study_design(path: Path = DEFAULT_SETTINGS_PATH) -> StudyDesign:
    """Load the study settings from a TOML file.

    Args:
        path: Location of the settings file.

    Returns:
        The validated study settings.

    Raises:
        ValueError: If the primary depth is not a student depth, or teacher tuning does not
            have the same number of candidates for every objective.
    """
    with path.open("rb") as stream:
        settings = tomllib.load(stream)
    design = StudyDesign(
        checkpoint_name=settings["checkpoint"]["name"],
        checkpoint_revision=settings["checkpoint"]["revision"],
        sessions=tuple(settings["corpus"]["sessions"]),
        emotions=tuple(settings["corpus"]["emotions"]),
        confirmatory_seeds=tuple(settings["seeds"]["confirmatory"]),
        tuning_seeds=tuple(settings["seeds"]["tuning"]),
        auxiliary_loss_scales=tuple(settings["teacher_tuning"]["auxiliary_loss_scales"]),
        asr_update_shares=tuple(settings["teacher_tuning"]["asr_update_shares"]),
        primary_depth=settings["student"]["primary_depth"],
        depths=tuple(settings["student"]["depths"]),
        training=TrainingSettings(**settings["training"]),
        distillation_temperature=settings["distillation"]["temperature"],
        distillation_candidates=tuple(
            DistillationCandidate(**candidate)
            for candidate in settings["distillation"]["candidates"]
        ),
    )
    if design.primary_depth not in design.depths:
        raise ValueError(f"Primary depth {design.primary_depth} is not in {design.depths}.")
    candidate_counts = {
        len(design.tuning_seeds),
        len(design.auxiliary_loss_scales),
        len(design.asr_update_shares),
    }
    if len(candidate_counts) != 1:
        raise ValueError("Teacher tuning needs the same number of candidates for every objective.")
    return design


def session_folds(sessions: tuple[int, ...]) -> tuple[SessionFold, ...]:
    """Build leave-one-session-out folds.

    Each session is held out once for testing. The next session, in cyclic order, is used for
    validation, and the remaining sessions are used for training.

    Args:
        sessions: Corpus session numbers in order.

    Returns:
        One fold for each session.
    """
    folds = []
    for position, test_session in enumerate(sessions):
        validation_session = sessions[(position + 1) % len(sessions)]
        training_sessions = tuple(
            session for session in sessions if session not in (test_session, validation_session)
        )
        folds.append(SessionFold(test_session, validation_session, training_sessions))
    return tuple(folds)


def _teacher_tuning_runs(design: StudyDesign, objective: TeacherObjective) -> list[RunSpec]:
    candidates = range(len(design.tuning_seeds))
    return [
        RunSpec(
            RunKind.TEACHER_TUNING,
            session,
            objective=objective,
            candidate=candidate,
            seed=(
                design.tuning_seeds[candidate]
                if objective is TeacherObjective.SER_ONLY
                else design.tuning_seeds[0]
            ),
        )
        for session in design.sessions
        for candidate in candidates
    ]


def _pilot_student_runs(design: StudyDesign, objective: TeacherObjective) -> list[RunSpec]:
    return [
        RunSpec(
            RunKind.PILOT_STUDENT,
            session,
            objective=objective,
            depth=design.primary_depth,
            seed=design.tuning_seeds[0],
        )
        for session in design.sessions
    ]


def _teacher_runs(design: StudyDesign, objective: TeacherObjective) -> list[RunSpec]:
    return [
        RunSpec(RunKind.TEACHER, session, objective=objective, seed=seed)
        for session in design.sessions
        for seed in design.confirmatory_seeds
    ]


def _distillation_tuning_runs(design: StudyDesign, objective: TeacherObjective) -> list[RunSpec]:
    return [
        RunSpec(
            RunKind.DISTILLATION_TUNING,
            session,
            objective=objective,
            depth=design.primary_depth,
            candidate=candidate,
            seed=design.tuning_seeds[0],
        )
        for session in design.sessions
        for candidate in range(len(design.distillation_candidates))
    ]


def _student_runs(
    design: StudyDesign,
    kind: RunKind,
    depths: tuple[int, ...],
    objective: TeacherObjective | None = None,
) -> list[RunSpec]:
    return [
        RunSpec(kind, session, objective=objective, depth=depth, seed=seed)
        for depth in depths
        for session in design.sessions
        for seed in design.confirmatory_seeds
    ]


def build_run_plan(design: StudyDesign, scope: PlanScope = PlanScope.FULL) -> tuple[RunSpec, ...]:
    """Build the ordered plan of training runs.

    Runs for the two core objectives come first, so that stopping early still leaves a complete
    primary comparison. Coefficient tuning always precedes the runs that use its result.

    Args:
        design: The study settings.
        scope: Whether to return the core comparison only or the full design.

    Returns:
        Run specifications in execution order.
    """
    primary = (design.primary_depth,)
    secondary = design.secondary_depths
    core_objectives = CORE_OBJECTIVES

    plan: list[RunSpec] = []
    for objective in core_objectives:
        plan += _teacher_tuning_runs(design, objective)
    for objective in core_objectives:
        plan += _pilot_student_runs(design, objective)
    if scope is PlanScope.PILOT:
        return tuple(plan)
    for objective in core_objectives:
        plan += _teacher_runs(design, objective)
    for objective in core_objectives:
        plan += _distillation_tuning_runs(design, objective)
    for objective in core_objectives:
        plan += _student_runs(design, RunKind.DISTILLED_STUDENT, primary, objective)
    for objective in core_objectives:
        plan += _student_runs(design, RunKind.INHERITED_STUDENT, primary, objective)
    plan += _student_runs(design, RunKind.BASE_STUDENT, primary)
    if scope is PlanScope.CORE:
        return tuple(plan)

    for objective in core_objectives:
        plan += _student_runs(design, RunKind.DISTILLED_STUDENT, secondary, objective)
    plan += _student_runs(design, RunKind.BASE_STUDENT, secondary)
    for objective in ADDITIONAL_OBJECTIVES:
        plan += _teacher_tuning_runs(design, objective)
        plan += _teacher_runs(design, objective)
    for objective in ADDITIONAL_OBJECTIVES:
        plan += _distillation_tuning_runs(design, objective)
        plan += _student_runs(design, RunKind.DISTILLED_STUDENT, primary, objective)
    for objective in ADDITIONAL_OBJECTIVES:
        plan += _student_runs(design, RunKind.DISTILLED_STUDENT, secondary, objective)
    return tuple(plan)
