import base64
import io
import json
import tarfile
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from compact_ser import iemocap_archive
from compact_ser.corpora import read_iemocap
from compact_ser.iemocap_archive import (
    CHECK_COMPLETION_MARKER,
    STORE_COMPLETION_MARKER,
    STUDY_ARCHIVE_NAME,
    STUDY_RECORD_NAME,
    DownloadError,
    build_study_archive,
    compute_checksums,
    copy_with_retries,
    download_archive,
    locate_study_store,
    main,
    restore_study_files,
    study_file_path,
)

USERNAME = "reader"
PASSWORD = "secret"
SESSIONS = (1, 2, 3, 4, 5)


@pytest.fixture(scope="module")
def release_archive(iemocap_root, tmp_path_factory):
    """A release-style archive with the study files and files the study does not use."""
    path = tmp_path_factory.mktemp("release") / "release.tar.gz"
    extra = {
        "IEMOCAP_full_release/README.txt": b"readme",
        "IEMOCAP_full_release/Session1/dialog/wav/Ses01F_impro01.wav": b"dialog audio",
        "IEMOCAP_full_release/Session1/dialog/EmoEvaluation/Categorical/x.txt": b"rating",
        "IEMOCAP_full_release/Session1/sentences/wav/Ses01F_impro01/._hidden.wav": b"hidden",
    }
    with tarfile.open(path, "w:gz") as archive:
        archive.add(iemocap_root, arcname="IEMOCAP_full_release")
        for name, content in extra.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return path


class ArchiveServer(BaseHTTPRequestHandler):
    """Serves one file with basic authentication and byte ranges."""

    content = b""
    redirect_to: str | None = None

    def do_GET(self):  # noqa: N802
        expected = base64.b64encode(f"{USERNAME}:{PASSWORD}".encode()).decode()
        if self.headers.get("Authorization") != f"Basic {expected}":
            self.send_response(401)
            self.end_headers()
            return
        if self.redirect_to:
            self.send_response(302)
            self.send_header("Location", self.redirect_to)
            self.end_headers()
            return
        start = 0
        if self.headers.get("Range"):
            start = int(self.headers["Range"].split("=")[1].rstrip("-"))
        body = self.content[start:]
        self.send_response(206 if start else 200)
        if start:
            self.send_header("Content-Range", f"bytes {start}-{len(self.content) - 1}/*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *arguments):
        pass


@pytest.fixture
def server(release_archive):
    ArchiveServer.content = release_archive.read_bytes()
    ArchiveServer.redirect_to = None
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), ArchiveServer)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}/release.tar.gz"
    httpd.shutdown()


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "IEMOCAP_full_release/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav",
            "Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav",
        ),
        (
            "IEMOCAP_full_release/Session2/dialog/EmoEvaluation/Ses02F_impro01.txt",
            "Session2/dialog/EmoEvaluation/Ses02F_impro01.txt",
        ),
        (
            "IEMOCAP_full_release/Session3/dialog/transcriptions/Ses03F_impro01.txt",
            "Session3/dialog/transcriptions/Ses03F_impro01.txt",
        ),
        ("IEMOCAP_full_release/Session1/dialog/wav/Ses01F_impro01.wav", None),
        ("IEMOCAP_full_release/Session1/dialog/EmoEvaluation/Categorical/x.txt", None),
        ("IEMOCAP_full_release/Session1/sentences/wav/Ses01F_impro01/._hidden.wav", None),
        ("IEMOCAP_full_release/README.txt", None),
    ],
)
def test_only_study_files_are_selected(name, expected):
    assert study_file_path(name) == expected


def test_study_archive_restores_the_same_corpus(release_archive, iemocap_root, tmp_path):
    record = build_study_archive(release_archive, tmp_path / STUDY_ARCHIVE_NAME)
    (tmp_path / STUDY_RECORD_NAME).write_text(json.dumps(record), encoding="utf-8")
    assert record["file_count"] == len(record["files"]) == 5 * 2 * (10 + 2)
    assert record["audio_files"] == 5 * 2 * 10

    corpus = restore_study_files(tmp_path, tmp_path / "runtime", log=lambda message: None)
    assert read_iemocap(corpus, SESSIONS) == read_iemocap(iemocap_root, SESSIONS)
    assert not (corpus / "README.txt").exists()


def test_missing_upload_lists_the_nearest_folder(tmp_path):
    (tmp_path / "compact-ser" / "datasets" / "iemocap-upload").mkdir(parents=True)
    store_dir = tmp_path / "compact-ser" / "datasets" / "iemocap"
    with pytest.raises(FileNotFoundError, match="datasets contains: iemocap-upload/"):
        restore_study_files(store_dir, tmp_path / "runtime", log=lambda message: None)


def test_stored_copy_is_found_in_another_folder(tmp_path):
    uploaded = tmp_path / "Research" / "compact-ser" / "datasets" / "iemocap"
    uploaded.mkdir(parents=True)
    for name in (STUDY_ARCHIVE_NAME, STUDY_RECORD_NAME):
        (uploaded / name).write_bytes(b"")
    (tmp_path / "compact-ser" / "runtime_check").mkdir(parents=True)
    expected = tmp_path / "compact-ser" / "datasets" / "iemocap"
    assert locate_study_store(expected, tmp_path, log=lambda message: None) == uploaded


def test_search_reports_when_no_copy_exists(tmp_path):
    (tmp_path / "compact-ser").mkdir()
    expected = tmp_path / "compact-ser" / "datasets" / "iemocap"
    with pytest.raises(FileNotFoundError, match="another Google account"):
        locate_study_store(expected, tmp_path, log=lambda message: None)


def test_restore_rejects_a_damaged_copy(release_archive, tmp_path):
    record = build_study_archive(release_archive, tmp_path / STUDY_ARCHIVE_NAME)
    record["archive_sha256"] = "0" * 64
    (tmp_path / STUDY_RECORD_NAME).write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        restore_study_files(tmp_path, tmp_path / "runtime", log=lambda message: None)


def test_truncated_release_is_rejected(release_archive, tmp_path):
    truncated = tmp_path / "truncated.tar.gz"
    content = release_archive.read_bytes()
    truncated.write_bytes(content[: len(content) - 20])
    with pytest.raises(ValueError, match="incomplete or damaged"):
        build_study_archive(truncated, tmp_path / STUDY_ARCHIVE_NAME)
    assert not (tmp_path / STUDY_ARCHIVE_NAME).exists()


def test_store_from_a_downloaded_archive(release_archive, iemocap_root, tmp_path, monkeypatch):
    counts = dict(Counter(u.emotion for u in read_iemocap(iemocap_root, SESSIONS)))
    monkeypatch.setattr(iemocap_archive, "IEMOCAP_FOUR_CLASS_COUNTS", counts)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    store_dir = tmp_path / "upload"
    main(
        [
            "store",
            "--source-archive",
            str(release_archive),
            "--work-dir",
            str(tmp_path / "work"),
            "--store-dir",
            str(store_dir),
            "--output-dir",
            str(tmp_path / "logs"),
        ]
    )
    record = json.loads((store_dir / STUDY_RECORD_NAME).read_text(encoding="utf-8"))
    assert record["source"]["expected_md5"] is None
    assert record["source"]["archive"] == release_archive.name


def test_download_requires_a_username_and_checksum(tmp_path):
    with pytest.raises(SystemExit):
        main(
            [
                "store",
                "--source-url",
                "https://example.org/a.tar.gz",
                "--work-dir",
                str(tmp_path),
                "--store-dir",
                str(tmp_path),
                "--output-dir",
                str(tmp_path),
            ]
        )


def test_unsafe_archive_paths_are_rejected(tmp_path):
    source = tmp_path / "unsafe.tar"
    with tarfile.open(source, "w") as archive:
        info = tarfile.TarInfo("../Session1/dialog/transcriptions/x.txt")
        archive.addfile(info, io.BytesIO(b""))
    with pytest.raises(ValueError, match="Unsafe"):
        build_study_archive(source, tmp_path / "out.tar")


def test_copy_resumes_from_a_partial_file(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(bytes(range(256)) * 100)
    destination = tmp_path / "copy" / "source.bin"
    destination.parent.mkdir()
    (destination.parent / "source.bin.part").write_bytes(source.read_bytes()[:1000])
    copy_with_retries(source, destination)
    assert destination.read_bytes() == source.read_bytes()
    assert not (destination.parent / "source.bin.part").exists()


def test_download_succeeds_and_resumes(server, release_archive, tmp_path):
    destination = tmp_path / "release.tar.gz"
    partial = tmp_path / "release.tar.gz.part"
    partial.write_bytes(release_archive.read_bytes()[:500])
    download_archive(server, USERNAME, PASSWORD, destination, log=lambda message: None)
    assert compute_checksums(destination) == compute_checksums(release_archive)
    assert not partial.exists()


def test_rejected_password_gives_a_clear_error(server, tmp_path):
    with pytest.raises(DownloadError, match="rejected the username or password"):
        download_archive(server, USERNAME, "wrong", tmp_path / "release.tar.gz")


def test_redirect_to_another_host_is_refused(server, tmp_path):
    ArchiveServer.redirect_to = "http://example.invalid/release.tar.gz"
    with pytest.raises(DownloadError, match="another host"):
        download_archive(server, USERNAME, PASSWORD, tmp_path / "release.tar.gz")


def test_store_and_check_commands(
    server, release_archive, iemocap_root, tmp_path, monkeypatch, capsys
):
    counts = dict(Counter(u.emotion for u in read_iemocap(iemocap_root, SESSIONS)))
    monkeypatch.setattr(iemocap_archive, "IEMOCAP_FOUR_CLASS_COUNTS", counts)
    monkeypatch.setattr("sys.stdin", io.StringIO(f"{PASSWORD}\n"))
    store_dir = tmp_path / "drive" / "datasets" / "iemocap"
    md5 = compute_checksums(release_archive)["md5"]
    main(
        [
            "store",
            "--source-url",
            server,
            "--username",
            USERNAME,
            "--expected-md5",
            md5,
            "--work-dir",
            str(tmp_path / "work"),
            "--store-dir",
            str(store_dir),
            "--output-dir",
            str(tmp_path / "store_logs"),
        ]
    )
    assert STORE_COMPLETION_MARKER in capsys.readouterr().out
    assert (store_dir / STUDY_ARCHIVE_NAME).exists()
    record = json.loads((store_dir / STUDY_RECORD_NAME).read_text(encoding="utf-8"))
    assert record["source"]["checksums"]["md5"] == md5

    main(
        [
            "check",
            "--store-dir",
            str(store_dir),
            "--local-dir",
            str(tmp_path / "runtime"),
            "--output-dir",
            str(tmp_path / "check_logs"),
        ]
    )
    assert CHECK_COMPLETION_MARKER in capsys.readouterr().out
    result = json.loads((tmp_path / "check_logs" / "iemocap_check.json").read_text("utf-8"))
    assert result["utterances"] == sum(counts.values())


def test_wrong_checksum_stops_before_anything_is_stored(server, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(f"{PASSWORD}\n"))
    store_dir = tmp_path / "drive"
    with pytest.raises(ValueError, match="expected MD5"):
        main(
            [
                "store",
                "--source-url",
                server,
                "--username",
                USERNAME,
                "--expected-md5",
                "0" * 32,
                "--work-dir",
                str(tmp_path / "work"),
                "--store-dir",
                str(store_dir),
                "--output-dir",
                str(tmp_path / "logs"),
            ]
        )
    assert "IEMOCAP STORE FAILED" in capsys.readouterr().out
    assert not store_dir.exists()
