import json
import time
from dataclasses import asdict, replace

import pytest

from compact_ser.run_adoption import adopt_runs, adoptable_run_id
from compact_ser.run_ledger import RunLedger, RunStatus
from compact_ser.study_design import load_study_design
from compact_ser.study_runner import (
    LEDGER_FILE_NAME,
    LOCK_FILE_NAME,
    MODEL_FILE_NAME,
    RESULT_FILE_NAME,
    SETTINGS_FILE_NAME,
)

DESIGN = load_study_design()
EMOTION_ONLY = "teacher_tuning__ser_only__session4__candidate1__seed20270202"
JOINT_AT_ONE = "teacher_tuning__ser_asr_sid__session1__candidate2__seed20270201"
JOINT_AT_TENTH = "teacher_tuning__ser_asr_sid__session1__candidate1__seed20270201"
UNFINISHED = "teacher_tuning__ser_only__session5__candidate0__seed20270201"


def pilot_settings(design):
    """The pilot's settings record: same teacher training, longer student training."""
    record = json.loads(json.dumps(asdict(design)))
    record["training"]["student_updates"] = 20000
    return record


def make_pilot(root, settings, runs):
    ledger = RunLedger(root / LEDGER_FILE_NAME)
    for run_id, scale, finished in runs:
        folder = root / "runs" / run_id
        folder.mkdir(parents=True)
        (folder / MODEL_FILE_NAME).write_bytes(b"weights")
        result = {"run_id": run_id, "wall_seconds": 60.0, "validation_metrics": {"uar": 0.7}}
        if scale is not None:
            result["auxiliary_loss_scale"] = scale
        ledger.record(run_id, RunStatus.STARTED)
        if finished:
            (folder / RESULT_FILE_NAME).write_text(json.dumps(result), encoding="utf-8")
            ledger.record(run_id, RunStatus.COMPLETED)
    (root / SETTINGS_FILE_NAME).write_text(json.dumps(settings), encoding="utf-8")


@pytest.fixture
def pilot(tmp_path):
    root = tmp_path / "study"
    make_pilot(
        root,
        pilot_settings(DESIGN),
        [
            (EMOTION_ONLY, None, True),
            (JOINT_AT_ONE, 1.0, True),
            (JOINT_AT_TENTH, 0.1, True),
            (UNFINISHED, None, False),
        ],
    )
    return root


def test_only_matching_completed_teachers_are_adopted(pilot, tmp_path):
    study = tmp_path / "main_study"
    adopted = adopt_runs(pilot, study, DESIGN, log=lambda message: None)
    assert adopted == [
        "teacher__ser_asr_sid__session1__seed20270201",
        "teacher__ser_only__session4__seed20270202",
    ]
    statuses = RunLedger(study / LEDGER_FILE_NAME).latest_statuses()
    assert set(statuses) == set(adopted)
    result = json.loads((study / "runs" / adopted[1] / RESULT_FILE_NAME).read_text("utf-8"))
    assert result["run_id"] == adopted[1]
    assert result["adopted_from"] == EMOTION_ONLY
    assert (study / "runs" / adopted[1] / MODEL_FILE_NAME).read_bytes() == b"weights"
    assert not (pilot / "runs" / EMOTION_ONLY).exists()
    assert (pilot / "runs" / JOINT_AT_TENTH).exists()


def test_adopting_again_changes_nothing(pilot, tmp_path):
    study = tmp_path / "main_study"
    adopt_runs(pilot, study, DESIGN, log=lambda message: None)
    assert adopt_runs(pilot, study, DESIGN, log=lambda message: None) == []


def test_teachers_trained_with_other_settings_are_refused(tmp_path):
    root = tmp_path / "study"
    settings = pilot_settings(DESIGN)
    settings["training"]["learning_rate"] = 1e-5
    make_pilot(root, settings, [(EMOTION_ONLY, None, True)])
    with pytest.raises(ValueError, match="other settings"):
        adopt_runs(root, tmp_path / "main_study", DESIGN, log=lambda message: None)


def test_a_running_source_is_refused(pilot, tmp_path):
    lock = {"owner": "other", "refreshed_at": time.time()}
    (pilot / LOCK_FILE_NAME).write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(RuntimeError, match="still training"):
        adopt_runs(pilot, tmp_path / "main_study", DESIGN, log=lambda message: None)


def test_a_missing_source_adopts_nothing(tmp_path):
    assert adopt_runs(tmp_path / "absent", tmp_path / "main", DESIGN, log=lambda m: None) == []


def test_seeds_outside_the_study_are_not_adopted():
    design = replace(DESIGN, seeds=(1, 2, 3))
    assert adoptable_run_id(EMOTION_ONLY, {}, design) is None
