import pytest

from compact_ser.run_ledger import RunLedger, RunStatus
from compact_ser.study_design import RunKind, RunSpec

FIRST_RUN = RunSpec(RunKind.BASE_STUDENT, test_session=1, depth=6, seed=1)
SECOND_RUN = RunSpec(RunKind.BASE_STUDENT, test_session=2, depth=6, seed=1)


def test_missing_ledger_means_every_run_is_pending(tmp_path):
    ledger = RunLedger(tmp_path / "ledger.jsonl")
    assert ledger.latest_statuses() == {}
    assert ledger.pending_runs([FIRST_RUN, SECOND_RUN]) == [FIRST_RUN, SECOND_RUN]


def test_only_completed_runs_are_skipped(tmp_path):
    ledger = RunLedger(tmp_path / "logs" / "ledger.jsonl")
    ledger.record(FIRST_RUN.run_id, RunStatus.STARTED)
    ledger.record(FIRST_RUN.run_id, RunStatus.COMPLETED, gpu_hours=1.5)
    ledger.record(SECOND_RUN.run_id, RunStatus.STARTED)
    assert ledger.pending_runs([FIRST_RUN, SECOND_RUN]) == [SECOND_RUN]


def test_a_failed_run_that_later_completes_is_not_pending(tmp_path):
    ledger = RunLedger(tmp_path / "ledger.jsonl")
    ledger.record(FIRST_RUN.run_id, RunStatus.FAILED)
    assert ledger.pending_runs([FIRST_RUN]) == [FIRST_RUN]
    ledger.record(FIRST_RUN.run_id, RunStatus.COMPLETED)
    assert ledger.pending_runs([FIRST_RUN]) == []


def test_partial_final_line_from_a_lost_session_is_ignored(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = RunLedger(path)
    ledger.record(FIRST_RUN.run_id, RunStatus.COMPLETED)
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"run_id": "partial')
    assert ledger.latest_statuses() == {FIRST_RUN.run_id: RunStatus.COMPLETED}


def test_damage_before_the_final_line_is_reported(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text('not json\n{"run_id": "a", "status": "completed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="damaged line"):
        RunLedger(path).latest_statuses()
