import json
import sys
from pathlib import Path

import pytest

from compact_ser import colab_jobs, iemocap_archive, runtime_check, training_job
from compact_ser.colab_jobs import (
    CODE_COMMIT_VARIABLE,
    DRIVE_ROOT,
    JOBS,
    LOCAL_DATA_DIR,
    ColabSession,
    find_job,
    follow_job,
    job_command,
    start_job,
)
from compact_ser.study_design import build_run_plan, load_study_design

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPOSITORY_ROOT / "notebooks" / "colab_runner.ipynb"


@pytest.fixture(scope="module")
def notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def code_cells(notebook):
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_job_markers_match_the_modules():
    expected = {
        "runtime-check": runtime_check.COMPLETION_MARKER,
        "iemocap-store": iemocap_archive.STORE_COMPLETION_MARKER,
        "iemocap-check": iemocap_archive.CHECK_COMPLETION_MARKER,
        "training": training_job.COMPLETION_MARKER,
    }
    assert {name: job.completion_marker for name, job in JOBS.items()} == expected
    assert JOBS["training"].failure_marker == training_job.FAILURE_MARKER
    assert JOBS["runtime-check"].failure_marker == runtime_check.FAILURE_MARKER


def test_unknown_job_names_the_choices():
    with pytest.raises(ValueError, match="iemocap-check, iemocap-store, runtime-check, training"):
        find_job("train")


def test_command_fills_places_and_keeps_user_options_last(tmp_path):
    session = ColabSession(JOBS["training"], tmp_path, REPOSITORY_ROOT, "abc123")
    command = job_command(session, ["--max-runs", "2"])
    assert command[1:4] == ["-u", "-m", "compact_ser.training_job"]
    assert command[command.index("--store-dir") + 1] == f"{tmp_path}/datasets/iemocap"
    assert command[command.index("--search-dir") + 1] == str(DRIVE_ROOT)
    assert command[command.index("--local-dir") + 1] == str(LOCAL_DATA_DIR)
    assert command[command.index("--output-dir") + 1] == str(tmp_path / "study")
    assert command[-2:] == ["--max-runs", "2"]


def test_job_runs_in_the_background_and_its_log_is_followed(tmp_path, capsys):
    session = ColabSession(JOBS["runtime-check"], tmp_path, REPOSITORY_ROOT, "abc123")
    options = ["--small-model", "--batch-size", "2", "--duration-seconds", "1"]
    start_job(session, [*options, "--updates", "1", "--warmup-updates", "0"])
    assert follow_job(session, interval=0.2)
    log = session.log_path.read_text(encoding="utf-8")
    assert log.startswith("--- ") and "at commit abc123" in log.splitlines()[0]
    assert "RUNTIME CHECK COMPLETE" in capsys.readouterr().out
    records = list((tmp_path / "runtime_check" / "environment").glob("*/session.json"))
    assert json.loads(records[0].read_text(encoding="utf-8"))["code_commit"] == "abc123"


def test_failed_job_is_reported_as_not_completed(tmp_path, capsys):
    session = ColabSession(JOBS["iemocap-check"], tmp_path, REPOSITORY_ROOT, "abc123")
    start_job(session, ["--local-dir", str(tmp_path / "runtime")])
    assert not follow_job(session, interval=0.2)
    assert "Job stopped before completing" in capsys.readouterr().out


def test_password_reaches_the_job_through_standard_input(tmp_path, monkeypatch):
    job = colab_jobs.ColabJob(
        module="compact_ser.colab_jobs",
        folder="echo",
        completion_marker="never",
        failure_marker="never",
        asks_for_password=True,
    )
    session = ColabSession(job, tmp_path, REPOSITORY_ROOT, "abc123")
    script = f"import os; print(input()[::-1], os.environ[{CODE_COMMIT_VARIABLE!r}])"
    monkeypatch.setattr(
        colab_jobs, "job_command", lambda session, options: [sys.executable, "-c", script]
    )
    start_job(session, password="secret")
    session.process.wait(timeout=30)
    assert "terces abc123" in session.log_path.read_text(encoding="utf-8")


def test_runner_notebook_is_valid_and_has_no_outputs(notebook):
    assert notebook["nbformat"] == 4
    identifiers = [cell["id"] for cell in notebook["cells"]]
    assert len(identifiers) == len(set(identifiers))
    for cell in code_cells(notebook):
        assert cell["outputs"] == [], "Clear the notebook's outputs before committing it."
        assert cell["execution_count"] is None
        compile("".join(cell["source"]), cell["id"], "exec")


def test_runner_notebook_defaults_to_a_known_job(notebook):
    parameters = code_cells(notebook)[0]
    assert parameters["metadata"]["tags"] == ["parameters"]
    namespace: dict[str, object] = {}
    exec("".join(parameters["source"]), namespace)
    assert namespace["JOB"] in JOBS
    assert namespace["DRIVE_FOLDER"] == "Research/compact-ser"
    options = namespace["JOB_OPTIONS"]
    if namespace["JOB"] == "training" and "--runs" in options:
        run_ids = options[options.index("--runs") + 1 :]
        training_job.select_runs(build_run_plan(load_study_design()), run_ids)
