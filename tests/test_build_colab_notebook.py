import base64
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = REPOSITORY_ROOT / "scripts" / "build_colab_notebook.py"


def load_builder():
    specification = importlib.util.spec_from_file_location("build_colab_notebook", BUILDER_PATH)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder():
    return load_builder()


@pytest.fixture(scope="module")
def notebook(builder):
    return builder.build_notebook(builder.JOBS["runtime-check"], builder.build_code_bundle())


def embedded_bundle(notebook) -> tuple[bytes, str]:
    namespace: dict[str, str] = {}
    exec("".join(notebook["cells"][1]["source"]), namespace)
    return base64.b64decode(namespace["BUNDLE_BASE64"]), namespace["BUNDLE_SHA256"]


def test_bundle_is_reproducible(builder):
    assert builder.build_code_bundle() == builder.build_code_bundle()


def test_bundle_contains_the_package_and_settings_only(builder):
    with zipfile.ZipFile(io.BytesIO(builder.build_code_bundle())) as archive:
        names = archive.namelist()
    assert "configs/study.toml" in names
    assert "src/compact_ser/runtime_check.py" in names
    assert all(name.startswith(("configs/", "src/compact_ser/")) for name in names)


def test_notebook_embeds_a_bundle_matching_its_checksum(notebook):
    bundle, checksum = embedded_bundle(notebook)
    assert hashlib.sha256(bundle).hexdigest() == checksum


def test_code_cells_are_valid_python(notebook):
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")


def test_notebook_is_valid_json_with_unique_cell_ids(notebook):
    json.dumps(notebook)
    identifiers = [cell["id"] for cell in notebook["cells"]]
    assert len(identifiers) == len(set(identifiers))
    assert notebook["nbformat"] == 4


def test_job_runs_from_the_extracted_bundle(notebook, tmp_path):
    bundle, _ = embedded_bundle(notebook)
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        archive.extractall(tmp_path / "code")
    environment = {**os.environ, "PYTHONPATH": str(tmp_path / "code" / "src")}
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "compact_ser.runtime_check",
            "--output-dir",
            str(tmp_path / "output"),
            "--small-model",
            "--batch-size",
            "2",
            "--duration-seconds",
            "1",
            "--updates",
            "1",
            "--warmup-updates",
            "0",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "RUNTIME CHECK COMPLETE" in completed.stdout


def test_builder_refuses_to_overwrite_a_notebook(tmp_path):
    command = [
        sys.executable,
        str(BUILDER_PATH),
        "--job",
        "runtime-check",
        "--tag",
        "test",
        "--output-dir",
        str(tmp_path),
    ]
    subprocess.run(command, check=True, capture_output=True)
    second = subprocess.run(command, check=False, capture_output=True, text=True)
    assert second.returncode != 0
    assert "exists" in second.stderr
