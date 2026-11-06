from collections import Counter

import pytest

from compact_ser.study_design import (
    ADDITIONAL_OBJECTIVES,
    CORE_OBJECTIVES,
    PlanScope,
    RunKind,
    TeacherObjective,
    build_run_plan,
    load_study_design,
    session_folds,
)


@pytest.fixture(scope="module")
def design():
    return load_study_design()


def test_every_session_is_tested_once_with_the_next_session_for_validation(design):
    folds = session_folds(design.sessions)
    assert [fold.test_session for fold in folds] == [1, 2, 3, 4, 5]
    assert [fold.validation_session for fold in folds] == [2, 3, 4, 5, 1]


def test_folds_keep_training_validation_and_test_sessions_separate(design):
    for fold in session_folds(design.sessions):
        assert len(fold.training_sessions) == 3
        assert fold.test_session not in fold.training_sessions
        assert fold.validation_session not in fold.training_sessions
        assert fold.test_session != fold.validation_session


def test_full_plan_has_the_expected_number_of_runs_of_each_kind(design):
    counts = Counter(run.kind for run in build_run_plan(design))
    assert counts == {
        RunKind.TEACHER_TUNING: 75,
        RunKind.PILOT_STUDENT: 10,
        RunKind.TEACHER: 75,
        RunKind.DISTILLATION_TUNING: 75,
        RunKind.DISTILLED_STUDENT: 150,
        RunKind.INHERITED_STUDENT: 30,
        RunKind.BASE_STUDENT: 30,
    }
    assert sum(counts.values()) == 445


def test_core_plan_has_the_expected_number_of_runs_of_each_kind(design):
    counts = Counter(run.kind for run in build_run_plan(design, PlanScope.CORE))
    assert counts == {
        RunKind.TEACHER_TUNING: 30,
        RunKind.PILOT_STUDENT: 10,
        RunKind.TEACHER: 30,
        RunKind.DISTILLATION_TUNING: 30,
        RunKind.DISTILLED_STUDENT: 30,
        RunKind.INHERITED_STUDENT: 30,
        RunKind.BASE_STUDENT: 15,
    }
    assert sum(counts.values()) == 175


def test_core_plan_is_the_start_of_the_full_plan(design):
    core = build_run_plan(design, PlanScope.CORE)
    full = build_run_plan(design)
    assert full[: len(core)] == core


def test_run_identifiers_are_unique_and_safe_as_folder_names(design):
    identifiers = [run.run_id for run in build_run_plan(design)]
    assert len(identifiers) == len(set(identifiers))
    assert all(identifier.replace("_", "").isalnum() for identifier in identifiers)


def test_primary_runs_of_an_objective_come_before_its_secondary_runs(design):
    plan = build_run_plan(design)
    for objective in (*CORE_OBJECTIVES, *ADDITIONAL_OBJECTIVES):
        positions = {
            run.depth: index
            for index, run in enumerate(plan)
            if run.kind is RunKind.DISTILLED_STUDENT and run.objective is objective
        }
        assert positions[design.primary_depth] < positions[4]


def test_tuning_precedes_the_runs_that_use_it(design):
    plan = build_run_plan(design)
    for objective in TeacherObjective:
        kinds = [run.kind for run in plan if run.objective is objective]
        assert kinds.index(RunKind.TEACHER_TUNING) < kinds.index(RunKind.TEACHER)
        assert kinds.index(RunKind.TEACHER) < kinds.index(RunKind.DISTILLATION_TUNING)
        assert kinds.index(RunKind.DISTILLATION_TUNING) < kinds.index(RunKind.DISTILLED_STUDENT)


def test_only_the_single_task_teacher_varies_its_seed_during_tuning(design):
    tuning = [run for run in build_run_plan(design) if run.kind is RunKind.TEACHER_TUNING]
    for objective in TeacherObjective:
        seeds = {run.seed for run in tuning if run.objective is objective}
        expected = 3 if objective is TeacherObjective.SER_ONLY else 1
        assert len(seeds) == expected
