import json
import tarfile
import time
from collections import Counter

import pytest

from compact_ser import iemocap_archive
from compact_ser.corpora import read_iemocap
from compact_ser.iemocap_archive import STUDY_ARCHIVE_NAME, STUDY_RECORD_NAME, build_study_archive
from compact_ser.run_ledger import RunLedger, RunStatus
from compact_ser.study_design import PlanScope, build_run_plan, load_study_design
from compact_ser.study_runner import LEDGER_FILE_NAME, RESULT_FILE_NAME
from compact_ser.training_job import (
    COMPLETION_MARKER,
    FAILURE_MARKER,
    LOCK_FILE_NAME,
    StudyLock,
    StudyLockedError,
    main,
    select_runs,
)

SESSIONS = (1, 2, 3, 4, 5)


@pytest.fixture
def store_dir(iemocap_root, tmp_path, monkeypatch):
    """A stored study archive of the synthetic corpus, as the store job leaves it on Drive."""
    release = tmp_path / "release.tar"
    with tarfile.open(release, "w") as archive:
        archive.add(iemocap_root, arcname="IEMOCAP_full_release")
    store = tmp_path / "drive" / "Research" / "compact-ser" / "datasets" / "iemocap"
    store.mkdir(parents=True)
    record = build_study_archive(release, store / STUDY_ARCHIVE_NAME)
    (store / STUDY_RECORD_NAME).write_text(json.dumps(record), encoding="utf-8")
    counts = dict(Counter(u.emotion for u in read_iemocap(iemocap_root, SESSIONS)))
    monkeypatch.setattr(iemocap_archive, "IEMOCAP_FOUR_CLASS_COUNTS", counts)
    return store


def job_arguments(store_dir, tmp_path, *options):
    return [
        "--store-dir",
        str(store_dir),
        "--local-dir",
        str(tmp_path / "runtime"),
        "--output-dir",
        str(tmp_path / "drive" / "study"),
        "--small-model",
        "--update-limit",
        "1",
        *options,
    ]


def test_job_restores_the_corpus_and_runs_the_next_pending_run(
    store_dir, tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("COMPACT_SER_CODE_COMMIT", "abc123")
    main(job_arguments(store_dir, tmp_path, "--max-runs", "1"))
    output = capsys.readouterr().out
    assert "1 runs executed; 174 selected runs still pending." in output
    assert COMPLETION_MARKER in output

    study = tmp_path / "drive" / "study"
    first = build_run_plan(load_study_design(), PlanScope.CORE)[0]
    assert RunLedger(study / LEDGER_FILE_NAME).latest_statuses() == {
        first.run_id: RunStatus.COMPLETED
    }
    result = json.loads((study / "runs" / first.run_id / RESULT_FILE_NAME).read_text("utf-8"))
    assert result["code_commit"] == "abc123"
    assert not (study / LOCK_FILE_NAME).exists()


def test_job_runs_only_the_named_runs(store_dir, tmp_path, capsys):
    plan = build_run_plan(load_study_design(), PlanScope.CORE)
    chosen = plan[4].run_id
    main(job_arguments(store_dir, tmp_path, "--runs", chosen))
    assert "1 runs executed; 0 selected runs still pending." in capsys.readouterr().out
    statuses = RunLedger(tmp_path / "drive" / "study" / LEDGER_FILE_NAME).latest_statuses()
    assert list(statuses) == [chosen]


def test_unknown_run_fails_before_training(store_dir, tmp_path, capsys):
    with pytest.raises(ValueError, match="not in the plan: no_such_run"):
        main(job_arguments(store_dir, tmp_path, "--runs", "no_such_run"))
    assert FAILURE_MARKER in capsys.readouterr().out


def test_select_runs_keeps_plan_order():
    plan = build_run_plan(load_study_design(), PlanScope.CORE)
    selected = select_runs(plan, [plan[3].run_id, plan[1].run_id])
    assert selected == [plan[1], plan[3]]


def test_recent_lock_from_another_runtime_stops_the_job(tmp_path):
    path = tmp_path / LOCK_FILE_NAME
    path.write_text(json.dumps({"owner": "other", "refreshed_at": time.time()}), "utf-8")
    with pytest.raises(StudyLockedError, match="Another runtime"), StudyLock(path):
        pass
    assert json.loads(path.read_text("utf-8"))["owner"] == "other"


def test_stale_lock_is_taken_over_and_released(tmp_path):
    path = tmp_path / LOCK_FILE_NAME
    path.write_text(json.dumps({"owner": "other", "refreshed_at": time.time() - 3600}), "utf-8")
    with StudyLock(path) as lock:
        assert json.loads(path.read_text("utf-8"))["owner"] == lock.owner
    assert not path.exists()


def test_lock_is_refreshed_while_held(tmp_path):
    path = tmp_path / LOCK_FILE_NAME
    with StudyLock(path, refresh_interval=0.05):
        first = json.loads(path.read_text("utf-8"))["refreshed_at"]
        time.sleep(0.3)
        assert json.loads(path.read_text("utf-8"))["refreshed_at"] > first
