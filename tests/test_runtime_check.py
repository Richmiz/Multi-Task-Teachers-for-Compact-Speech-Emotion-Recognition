import json

import pytest

from compact_ser.runtime_check import COMPLETION_MARKER, RESULT_FILE_NAME, main

SMALL_RUN = [
    "--small-model",
    "--batch-size",
    "2",
    "--duration-seconds",
    "1",
    "--updates",
    "2",
    "--warmup-updates",
    "1",
]


def test_check_saves_measurements_and_an_environment_record(tmp_path, capsys):
    main(["--output-dir", str(tmp_path), *SMALL_RUN])

    results = json.loads((tmp_path / RESULT_FILE_NAME).read_text(encoding="utf-8"))
    assert set(results["measurements"]) == {
        "teacher_ser_asr_sid",
        "student_depth6",
        "student_depth4",
    }
    for measurement in results["measurements"].values():
        assert measurement["seconds_per_update"] > 0

    record = tmp_path / "environment" / results["environment_record"]
    assert (record / "session.json").exists()
    assert (record / "nvidia_smi.txt").exists()
    assert (record / "pip_freeze.txt").exists()
    assert COMPLETION_MARKER in capsys.readouterr().out


def test_failed_check_prints_the_failure_marker(tmp_path, capsys):
    settings = [*SMALL_RUN[:-4], "--updates", "0", "--warmup-updates", "0"]
    with pytest.raises(ValueError):
        main(["--output-dir", str(tmp_path), *settings])
    assert "RUNTIME CHECK FAILED" in capsys.readouterr().out
