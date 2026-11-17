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

    TEACHER = "teacher"
    DISTILLED_STUDENT = "distilled_student"
    INHERITED_STUDENT = "inherited_student"
    BASE_STUDENT = "base_student"


class PlanScope(StrEnum):
    """Parts of the study that a run plan can cover.

    `CORE` is the primary comparison and its controls; `FULL` adds the other teacher objectives
    and the secondary student depth.
    """

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
    mixed_precision: bool
    group_batches_by_length: bool


@dataclass(frozen=True)
class DistillationWeights:
    """Weights of the distillation loss terms."""

    logit_weight: float
    hidden_state_weight: float


@dataclass(frozen=True)
class StudyDesign:
    """Fixed study settings loaded from the configuration file."""

    checkpoint_name: str
    checkpoint_revision: str
    sessions: tuple[int, ...]
    emotions: tuple[str, ...]
    seeds: tuple[int, ...]
    auxiliary_loss_scale: float
    asr_update_share: float
    primary_depth: int
    depths: tuple[int, ...]
    training: TrainingSettings
    distillation_temperature: float
    distillation: DistillationWeights

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
        ValueError: If the primary depth is not a student depth.
    """
    with path.open("rb") as stream:
        settings = tomllib.load(stream)
    distillation = settings["distillation"]
    design = StudyDesign(
        checkpoint_name=settings["checkpoint"]["name"],
        checkpoint_revision=settings["checkpoint"]["revision"],
        sessions=tuple(settings["corpus"]["sessions"]),
        emotions=tuple(settings["corpus"]["emotions"]),
        seeds=tuple(settings["seeds"]["replicates"]),
        auxiliary_loss_scale=settings["teachers"]["auxiliary_loss_scale"],
        asr_update_share=settings["teachers"]["asr_update_share"],
        primary_depth=settings["student"]["primary_depth"],
        depths=tuple(settings["student"]["depths"]),
        training=TrainingSettings(**settings["training"]),
        distillation_temperature=distillation["temperature"],
        distillation=DistillationWeights(
            logit_weight=distillation["logit_weight"],
            hidden_state_weight=distillation["hidden_state_weight"],
        ),
    )
    if design.primary_depth not in design.depths:
        raise ValueError(f"Primary depth {design.primary_depth} is not in {design.depths}.")
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


def _teacher_runs(design: StudyDesign, objective: TeacherObjective) -> list[RunSpec]:
    return [
        RunSpec(RunKind.TEACHER, session, objective=objective, seed=seed)
        for session in design.sessions
        for seed in design.seeds
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
        for seed in design.seeds
    ]


def build_run_plan(design: StudyDesign, scope: PlanScope = PlanScope.FULL) -> tuple[RunSpec, ...]:
    """Build the ordered plan of training runs.

    Runs for the two core objectives come first, so that stopping early still leaves a complete
    primary comparison. Teachers always precede the students that learn from them.

    Args:
        design: The study settings.
        scope: Whether to return the core comparison only or the full design.

    Returns:
        Run specifications in execution order.
    """
    primary = (design.primary_depth,)
    secondary = design.secondary_depths

    plan: list[RunSpec] = []
    for objective in CORE_OBJECTIVES:
        plan += _teacher_runs(design, objective)
    for objective in CORE_OBJECTIVES:
        plan += _student_runs(design, RunKind.DISTILLED_STUDENT, primary, objective)
    for objective in CORE_OBJECTIVES:
        plan += _student_runs(design, RunKind.INHERITED_STUDENT, primary, objective)
    plan += _student_runs(design, RunKind.BASE_STUDENT, primary)
    if scope is PlanScope.CORE:
        return tuple(plan)

    for objective in CORE_OBJECTIVES:
        plan += _student_runs(design, RunKind.DISTILLED_STUDENT, secondary, objective)
    plan += _student_runs(design, RunKind.BASE_STUDENT, secondary)
    for objective in ADDITIONAL_OBJECTIVES:
        plan += _teacher_runs(design, objective)
        plan += _student_runs(design, RunKind.DISTILLED_STUDENT, primary, objective)
    for objective in ADDITIONAL_OBJECTIVES:
        plan += _student_runs(design, RunKind.DISTILLED_STUDENT, secondary, objective)
    return tuple(plan)
