import json
import shutil
from dataclasses import replace

import pytest
import torch

from compact_ser.corpora import read_iemocap, write_manifest
from compact_ser.run_ledger import RunLedger, RunStatus
from compact_ser.study_design import (
    PlanScope,
    RunKind,
    TeacherObjective,
    build_run_plan,
    load_study_design,
)
from compact_ser.study_runner import (
    COMPLETION_MARKER,
    MODEL_FILE_NAME,
    RESULT_FILE_NAME,
    StudyContext,
    _small_encoder,
    execute_pending_runs,
    main,
)

TEST_SESSION = 1


@pytest.fixture(scope="module")
def design():
    base = load_study_design()
    training = replace(
        base.training,
        batch_size=4,
        teacher_updates=4,
        student_updates=2,
        validation_interval=1,
        gradient_log_interval=1,
        calibration_batches=1,
    )
    return replace(base, confirmatory_seeds=base.confirmatory_seeds[:1], training=training)


@pytest.fixture(scope="module")
def sealed_corpus(iemocap_root, tmp_path_factory):
    """Copy the corpus without the test session's audio, so reading it would fail."""
    root = tmp_path_factory.mktemp("sealed") / "iemocap"
    shutil.copytree(iemocap_root, root)
    shutil.rmtree(root / f"Session{TEST_SESSION}" / "sentences")
    return root


@pytest.fixture(scope="module")
def manifest(iemocap_root):
    return read_iemocap(iemocap_root, (1, 2, 3, 4, 5))


def make_context(design, manifest, corpus_root, output_root):
    return StudyContext(
        design=design,
        manifest=manifest,
        corpus_root=corpus_root,
        output_root=output_root,
        device=torch.device("cpu"),
        encoder_factory=_small_encoder,
    )


@pytest.fixture(scope="module")
def completed_study(design, manifest, sealed_corpus, tmp_path_factory):
    output_root = tmp_path_factory.mktemp("outputs")
    plan = [run for run in build_run_plan(design) if run.test_session == TEST_SESSION]
    context = make_context(design, manifest, sealed_corpus, output_root)
    executed = execute_pending_runs(context, plan)
    return context, plan, executed


def read_result(context, run):
    return json.loads((context.run_folder(run.run_id) / RESULT_FILE_NAME).read_text("utf-8"))


def test_full_plan_for_one_fold_runs_without_reading_the_test_session(completed_study):
    _, plan, executed = completed_study
    assert executed == len(plan) == 51


def test_every_run_saves_a_model_and_result_and_is_recorded(completed_study):
    context, plan, _ = completed_study
    statuses = RunLedger(context.output_root / "run_ledger.jsonl").latest_statuses()
    for run in plan:
        assert (context.run_folder(run.run_id) / MODEL_FILE_NAME).exists()
        assert statuses[run.run_id] is RunStatus.COMPLETED
        result = read_result(context, run)
        assert 0.0 <= result["validation_metrics"]["uar"] <= 1.0


def test_completed_runs_are_skipped_on_restart(completed_study):
    context, plan, _ = completed_study
    assert execute_pending_runs(context, plan) == 0


@pytest.mark.parametrize(
    ("objective", "expected_terms"),
    [
        (TeacherObjective.SER_ONLY, {"emotion"}),
        (TeacherObjective.SER_ASR_SID, {"emotion", "transcript", "speaker"}),
        (TeacherObjective.SER_ASR, {"emotion", "transcript"}),
        (TeacherObjective.SER_SID, {"emotion", "speaker"}),
        (TeacherObjective.ASR_THEN_SER, {"emotion"}),
    ],
)
def test_each_objective_trains_only_its_own_tasks(completed_study, objective, expected_terms):
    context, plan, _ = completed_study
    teachers = [
        run for run in plan if run.kind is RunKind.TEACHER_TUNING and run.objective is objective
    ]
    assert len(teachers) == 3
    for run in teachers:
        entry = read_result(context, run)["gradient_history"][0]
        shares = {name for name in entry if name.endswith("_gradient_share")}
        assert shares == {f"{term}_gradient_share" for term in expected_terms}
        if "speaker" in expected_terms:
            assert "speaker_training_accuracy" in entry


def test_sequential_teacher_splits_updates_between_stages(completed_study):
    context, plan, _ = completed_study
    runs = [
        run
        for run in plan
        if run.kind is RunKind.TEACHER_TUNING and run.objective is TeacherObjective.ASR_THEN_SER
    ]
    results = [read_result(context, run) for run in runs]
    assert [result["asr_update_share"] for result in results] == [0.25, 0.5, 0.75]
    assert [result["validation_history"][-1]["update"] for result in results] == [3, 2, 1]
    entry = results[0]["adaptation_gradient_history"][0]
    assert entry["transcript_gradient_share"] == pytest.approx(1.0)


def test_confirmatory_teacher_uses_the_best_tuning_candidate(completed_study):
    context, plan, _ = completed_study
    objective = TeacherObjective.SER_ASR_SID
    tuning = [
        read_result(context, run)
        for run in plan
        if run.kind is RunKind.TEACHER_TUNING and run.objective is objective
    ]
    scores = [result["validation_metrics"]["uar"] for result in tuning]
    teacher = next(
        run for run in plan if run.kind is RunKind.TEACHER and run.objective is objective
    )
    assert read_result(context, teacher)["candidate"] == scores.index(max(scores))


def test_command_line_run_prints_the_completion_marker(manifest, iemocap_root, tmp_path, capsys):
    manifest_path = tmp_path / "iemocap.jsonl"
    write_manifest(manifest, manifest_path)
    arguments = [
        "--manifest",
        str(manifest_path),
        "--corpus-root",
        str(iemocap_root),
        "--output-root",
        str(tmp_path / "outputs"),
        "--small-model",
        "--update-limit",
        "1",
        "--max-runs",
        "1",
    ]
    main(arguments)
    output = capsys.readouterr().out
    assert "1 runs executed." in output
    assert COMPLETION_MARKER in output


def test_multitask_teacher_learns_six_training_speakers_and_reports_transcription(completed_study):
    context, plan, _ = completed_study
    run = next(
        run
        for run in plan
        if run.kind is RunKind.TEACHER and run.objective is TeacherObjective.SER_ASR_SID
    )
    result = read_result(context, run)
    assert result["architecture"]["num_speakers"] == 6
    assert "word_error_rate" in result["validation_metrics"]
    assert "auxiliary_loss_scale" in result


def test_student_types_use_the_expected_teacher_and_losses(completed_study):
    context, plan, _ = completed_study
    for run in plan:
        if run.kind not in (
            RunKind.DISTILLED_STUDENT,
            RunKind.INHERITED_STUDENT,
            RunKind.BASE_STUDENT,
        ):
            continue
        result = read_result(context, run)
        terms = set(result["gradient_history"][0]) - {"update"}
        if run.kind is RunKind.DISTILLED_STUDENT:
            assert result["source_teacher"].startswith(f"teacher__{run.objective.value}")
            assert terms == {
                "emotion_gradient_share",
                "soft_targets_gradient_share",
                "hidden_states_gradient_share",
            }
        elif run.kind is RunKind.INHERITED_STUDENT:
            assert result["source_teacher"].startswith(f"teacher__{run.objective.value}")
            assert terms == {"emotion_gradient_share"}
        else:
            assert result["source_teacher"] is None
            assert terms == {"emotion_gradient_share"}


def test_student_keeps_only_the_emotion_head(completed_study):
    context, plan, _ = completed_study
    run = next(run for run in plan if run.kind is RunKind.DISTILLED_STUDENT)
    state = torch.load(
        context.run_folder(run.run_id) / MODEL_FILE_NAME, map_location="cpu", weights_only=True
    )
    assert any(name.startswith("emotion_head") for name in state)
    assert not any(name.startswith(("transcript_head", "speaker_head")) for name in state)
    layers = {name.split(".")[3] for name in state if name.startswith("encoder.encoder.layers.")}
    assert len(layers) == run.depth


def test_frontend_stays_frozen_in_trained_students(completed_study, design):
    context, plan, _ = completed_study
    run = next(run for run in plan if run.kind is RunKind.BASE_STUDENT)
    state = torch.load(
        context.run_folder(run.run_id) / MODEL_FILE_NAME, map_location="cpu", weights_only=True
    )
    source = _small_encoder().state_dict()
    name = "feature_extractor.conv_layers.0.conv.weight"
    assert torch.equal(state[f"encoder.{name}"], source[name])


def test_failed_run_is_recorded_and_stops_the_plan(design, manifest, tmp_path):
    missing_corpus = tmp_path / "missing"
    plan = build_run_plan(design, PlanScope.CORE)[:1]
    context = make_context(design, manifest, missing_corpus, tmp_path / "outputs")
    with pytest.raises(Exception, match="missing|No such file|System error|Error"):
        execute_pending_runs(context, plan)
    statuses = RunLedger(context.output_root / "run_ledger.jsonl").latest_statuses()
    assert statuses[plan[0].run_id] is RunStatus.FAILED


def test_student_cannot_start_before_its_teacher(design, manifest, iemocap_root, tmp_path):
    plan = [
        run
        for run in build_run_plan(design, PlanScope.CORE)
        if run.kind is RunKind.INHERITED_STUDENT
    ][:1]
    context = make_context(design, manifest, iemocap_root, tmp_path / "outputs")
    with pytest.raises(FileNotFoundError, match="must complete"):
        execute_pending_runs(context, plan)
