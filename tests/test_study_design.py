from collections import Counter

import pytest

from compact_ser.study_design import (
    ADDITIONAL_OBJECTIVES,
    CORE_OBJECTIVES,
    PlanScope,
    RunKind,
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


def test_core_plan_has_the_expected_number_of_runs_of_each_kind(design):
    counts = Counter(run.kind for run in build_run_plan(design, PlanScope.CORE))
    assert counts == {
        RunKind.TEACHER: 30,
        RunKind.DISTILLED_STUDENT: 30,
        RunKind.INHERITED_STUDENT: 30,
        RunKind.BASE_STUDENT: 15,
    }


def test_full_plan_has_the_expected_number_of_runs_of_each_kind(design):
    counts = Counter(run.kind for run in build_run_plan(design))
    assert counts == {
        RunKind.TEACHER: 75,
        RunKind.DISTILLED_STUDENT: 150,
        RunKind.INHERITED_STUDENT: 30,
        RunKind.BASE_STUDENT: 30,
    }


def test_core_plan_is_the_start_of_the_full_plan(design):
    core = build_run_plan(design, PlanScope.CORE)
    assert build_run_plan(design)[: len(core)] == core


def test_every_fold_and_seed_has_one_teacher_per_core_objective(design):
    teachers = [
        run for run in build_run_plan(design, PlanScope.CORE) if run.kind is RunKind.TEACHER
    ]
    expected = {
        (objective, session, seed)
        for objective in CORE_OBJECTIVES
        for session in design.sessions
        for seed in design.seeds
    }
    assert {(run.objective, run.test_session, run.seed) for run in teachers} == expected


def test_teachers_precede_the_students_that_learn_from_them(design):
    plan = build_run_plan(design)
    positions = {run.run_id: index for index, run in enumerate(plan)}
    for index, run in enumerate(plan):
        if run.kind in (RunKind.DISTILLED_STUDENT, RunKind.INHERITED_STUDENT):
            teacher = next(
                other
                for other in plan
                if other.kind is RunKind.TEACHER
                and (other.objective, other.test_session, other.seed)
                == (run.objective, run.test_session, run.seed)
            )
            assert positions[teacher.run_id] < index


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
