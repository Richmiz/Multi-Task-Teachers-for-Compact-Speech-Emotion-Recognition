"""Store the IEMOCAP files used by the study on Google Drive, and restore them on a runtime.

The licensed archive is downloaded once with a temporary credential. Only the utterance audio,
emotion evaluations, and transcripts are kept, in a smaller archive that each later session copies
from Drive and verifies before use. Copies are made in chunks and resumed after interruptions,
because Google Drive connections can drop during large transfers.

    python -m compact_ser.iemocap_archive store --source-url URL --username NAME \
        --expected-md5 MD5 --work-dir /content/iemocap-download \
        --store-dir /content/drive/MyDrive/Research/compact-ser/datasets/iemocap --output-dir LOGS
    python -m compact_ser.iemocap_archive check \
        --store-dir /content/drive/MyDrive/Research/compact-ser/datasets/iemocap \
        --local-dir /content/compact-ser-data --output-dir LOGS

The download password is read from standard input, so it is never written to disk or shown. An
archive that was already downloaded can be used with `--source-archive PATH` instead; the stored
copy can then be built on a local computer and its two files uploaded to the same Drive folder.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from compact_ser.corpora import IEMOCAP_FOUR_CLASS_COUNTS, read_iemocap

STUDY_ARCHIVE_NAME = "iemocap_study_files.tar"
STUDY_RECORD_NAME = "iemocap_study_files.json"
CHECK_RESULT_NAME = "iemocap_check.json"
SESSIONS = (1, 2, 3, 4, 5)
SESSION_FOLDERS = {f"Session{session}" for session in SESSIONS}
CHUNK_SIZE = 8 * 1024 * 1024
STORE_COMPLETION_MARKER = "IEMOCAP STORE COMPLETE"
STORE_FAILURE_MARKER = "IEMOCAP STORE FAILED"
CHECK_COMPLETION_MARKER = "IEMOCAP CHECK COMPLETE"
CHECK_FAILURE_MARKER = "IEMOCAP CHECK FAILED"

Logger = Callable[[str], None]


class DownloadError(RuntimeError):
    """Raised when the licensed archive cannot be downloaded."""


def _log(message: str) -> None:
    print(message, flush=True)


def _origin(url: str) -> tuple[str, str]:
    parts = urllib.parse.urlsplit(url)
    return parts.scheme, parts.netloc


class _SameOriginRedirect(urllib.request.HTTPRedirectHandler):
    """Follow redirects only within the source server, so the credential stays there."""

    def __init__(self, origin: tuple[str, str]) -> None:
        super().__init__()
        self.origin = origin

    def redirect_request(self, request, fp, code, message, headers, new_url):  # noqa: D102
        if _origin(new_url) != self.origin:
            raise DownloadError(
                "The server redirected to another host. The credential was not sent."
            )
        return super().redirect_request(request, fp, code, message, headers, new_url)


def compute_checksums(path: Path) -> dict[str, str]:
    """Compute the MD5 and SHA-256 checksums of a file.

    Args:
        path: The file to read.

    Returns:
        The hexadecimal checksums by algorithm name.
    """
    digests = {"md5": hashlib.md5(), "sha256": hashlib.sha256()}
    with path.open("rb") as stream:
        while block := stream.read(CHUNK_SIZE):
            for digest in digests.values():
                digest.update(block)
    return {name: digest.hexdigest() for name, digest in digests.items()}


def download_archive(
    url: str,
    username: str,
    password: str,
    destination: Path,
    attempts: int = 5,
    retry_delay: float = 10.0,
    report_interval: float = 30.0,
    log: Logger = _log,
) -> None:
    """Download a file with HTTP basic authentication, resuming after interruptions.

    The data are written to a `.part` file, which is renamed only after the transfer completes. A
    `.part` file left by an earlier attempt is resumed.

    Args:
        url: Address of the file.
        username: Username for HTTP basic authentication.
        password: Password for HTTP basic authentication.
        destination: Where to save the file.
        attempts: Number of connection attempts.
        retry_delay: Seconds to wait before reconnecting.
        report_interval: Seconds between progress messages.
        log: Receives progress messages.

    Raises:
        DownloadError: If the server rejects the credential, returns an error, or the transfer
            cannot be completed.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    token = base64.b64encode(f"{username}:{password}".encode()).decode("ascii")
    opener = urllib.request.build_opener(_SameOriginRedirect(_origin(url)))
    for attempt in range(1, attempts + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Authorization": f"Basic {token}", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with opener.open(urllib.request.Request(url, headers=headers), timeout=60) as response:
                resumed = offset > 0 and response.status == 206
                if resumed and not response.headers.get("Content-Range", "").startswith(
                    f"bytes {offset}-"
                ):
                    raise DownloadError("The server returned an inconsistent resume range.")
                if not resumed:
                    offset = 0
                length = response.headers.get("Content-Length")
                total = offset + int(length) if length is not None else None
                if total is not None and shutil.disk_usage(partial.parent).free < total - offset:
                    raise DownloadError(f"Not enough free disk space in {partial.parent}.")
                if total is not None:
                    size, start = total / 2**30, offset / 2**30
                    log(f"Archive size {size:.2f} GiB; starting at {start:.2f} GiB.")
                written = offset
                started = last_report = time.monotonic()
                with partial.open("ab" if resumed else "wb") as output:
                    while block := response.read(1024 * 1024):
                        output.write(block)
                        written += len(block)
                        now = time.monotonic()
                        if now - last_report >= report_interval:
                            speed = (written - offset) / max(now - started, 1e-3) / 2**20
                            share = f" of {total / 2**30:.2f} GiB" if total else ""
                            log(f"Downloaded {written / 2**30:.2f} GiB{share}, {speed:.1f} MiB/s.")
                            last_report = now
                if total is not None and written != total:
                    raise OSError("The transfer ended before the whole file arrived.")
            partial.replace(destination)
            return
        except urllib.error.HTTPError as error:
            if error.code in (401, 403):
                raise DownloadError(
                    f"The server rejected the username or password (HTTP {error.code}). "
                    "Check that the temporary credential is current and was typed correctly."
                ) from error
            if error.code == 416 and offset:
                # The partial file already holds every byte; the checksum check confirms it.
                partial.replace(destination)
                return
            raise DownloadError(f"The server returned HTTP {error.code}.") from error
        except (OSError, TimeoutError) as error:
            log(
                f"Connection interrupted ({type(error).__name__}); attempt {attempt} of {attempts}."
            )
            if attempt == attempts:
                raise DownloadError(
                    "The download did not finish. Run it again to resume from the partial file."
                ) from error
            time.sleep(retry_delay)


def study_file_path(member_name: str) -> str | None:
    """Return the corpus path of an archive member that the study uses.

    The study uses utterance audio, emotion evaluation files, and transcripts. Subfolders of the
    evaluation folder, whole-dialogue audio, and hidden files are not used.

    Args:
        member_name: Path of the member in the release archive.

    Returns:
        The path relative to the corpus folder, such as `Session1/sentences/wav/...`, or `None`
        if the study does not use the member.
    """
    path = PurePosixPath(member_name)
    if path.name.startswith("."):
        return None
    index = next((i for i, part in enumerate(path.parts) if part in SESSION_FOLDERS), None)
    if index is None:
        return None
    relative = path.parts[index + 1 :]
    is_audio = len(relative) == 4 and relative[:2] == ("sentences", "wav") and path.suffix == ".wav"
    is_text = (
        len(relative) == 3
        and relative[0] == "dialog"
        and relative[1] in ("EmoEvaluation", "transcriptions")
        and path.suffix == ".txt"
    )
    if not (is_audio or is_text):
        return None
    return PurePosixPath(*path.parts[index:]).as_posix()


def _open_release(source: Path) -> BinaryIO:
    with source.open("rb") as probe:
        is_gzip = probe.read(2) == b"\x1f\x8b"
    return gzip.open(source, "rb") if is_gzip else source.open("rb")


def build_study_archive(source: Path, destination: Path) -> dict[str, Any]:
    """Copy the files the study uses from the release archive into a smaller archive.

    A gzip-compressed release is read to its end, so its length and CRC checksum are verified.

    Args:
        source: The release archive, gzip-compressed or not.
        destination: The archive to create.

    Returns:
        A record with the archive checksum, file counts, and each file's SHA-256 checksum.

    Raises:
        ValueError: If the release archive is incomplete or damaged, or contains an unsafe or
            duplicate path.
    """
    files: dict[str, str] = {}
    partial = destination.with_name(destination.name + ".part")
    try:
        with (
            _open_release(source) as stream,
            tarfile.open(fileobj=stream, mode="r|", bufsize=CHUNK_SIZE) as archive,
            tarfile.open(partial, "w", format=tarfile.PAX_FORMAT) as output,
        ):
            for member in archive:
                name = PurePosixPath(member.name)
                if name.is_absolute() or ".." in name.parts:
                    raise ValueError(f"Unsafe path in the archive: {member.name}")
                relative = study_file_path(member.name)
                if relative is None or not member.isfile():
                    continue
                if relative in files:
                    raise ValueError(f"The archive contains {relative} more than once.")
                content = archive.extractfile(member).read()
                info = tarfile.TarInfo(relative)
                info.size = len(content)
                info.mode = 0o644
                output.addfile(info, io.BytesIO(content))
                files[relative] = hashlib.sha256(content).hexdigest()
            while stream.read(CHUNK_SIZE):
                pass
    except (EOFError, gzip.BadGzipFile, zlib.error, tarfile.ReadError) as error:
        raise ValueError(f"{source.name} is incomplete or damaged: {error}") from error
    partial.replace(destination)
    return {
        "archive": destination.name,
        "archive_sha256": compute_checksums(destination)["sha256"],
        "file_count": len(files),
        "audio_files": sum(name.endswith(".wav") for name in files),
        "files_by_session": dict(sorted(Counter(name.split("/")[0] for name in files).items())),
        "files": files,
    }


def copy_with_retries(
    source: Path,
    destination: Path,
    attempts: int = 5,
    retry_delay: float = 10.0,
    log: Logger = _log,
) -> None:
    """Copy a file in chunks, resuming from a `.part` file after an interruption.

    Args:
        source: The file to copy.
        destination: Where to write the copy.
        attempts: Number of attempts.
        retry_delay: Seconds to wait before retrying.
        log: Receives messages about retries.

    Raises:
        OSError: If the copy still fails after every attempt.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    size = source.stat().st_size
    for attempt in range(1, attempts + 1):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > size:
                partial.unlink()
                offset = 0
            with source.open("rb") as reader, partial.open("ab") as writer:
                reader.seek(offset)
                while block := reader.read(CHUNK_SIZE):
                    writer.write(block)
                writer.flush()
                os.fsync(writer.fileno())
            if partial.stat().st_size != size:
                raise OSError(f"Copied {partial.stat().st_size} of {size} bytes.")
            partial.replace(destination)
            return
        except OSError as error:
            log(f"Copy of {source.name} interrupted ({error}); attempt {attempt} of {attempts}.")
            if attempt == attempts:
                raise
            time.sleep(retry_delay)


def _sha256_with_retries(path: Path, attempts: int = 5, retry_delay: float = 10.0) -> str:
    for attempt in range(1, attempts + 1):
        try:
            return compute_checksums(path)["sha256"]
        except OSError:
            if attempt == attempts:
                raise
            time.sleep(retry_delay)
    raise AssertionError("unreachable")


def _describe_missing(path: Path, limit: int = 30) -> str:
    """Say which file is missing and what the nearest existing folder holds."""
    folder = path.parent
    while not folder.is_dir() and folder != folder.parent:
        folder = folder.parent
    entries = sorted(f"{item.name}/" if item.is_dir() else item.name for item in folder.iterdir())
    listing = ", ".join(entries[:limit]) if entries else "nothing"
    if len(entries) > limit:
        listing += f", and {len(entries) - limit} more"
    return f"{path} was not found. The folder {folder} contains: {listing}."


def locate_study_store(
    store_dir: Path,
    search_dir: Path | None = None,
    max_depth: int = 6,
    log: Logger = _log,
) -> Path:
    """Find the folder that holds the stored study archive and its record.

    The expected folder is used when it holds both files. Otherwise the folders under
    `search_dir` are searched breadth first, so a copy uploaded to another folder is still found.

    Args:
        store_dir: The folder where the files are expected.
        search_dir: Folder to search when `store_dir` does not hold the files.
        max_depth: Deepest folder level below `search_dir` that is searched.
        log: Receives progress messages.

    Returns:
        The folder holding both files.

    Raises:
        FileNotFoundError: If no searched folder holds both files.
    """
    wanted = {STUDY_RECORD_NAME, STUDY_ARCHIVE_NAME}
    if all((store_dir / name).is_file() for name in wanted):
        return store_dir
    missing = _describe_missing(store_dir / STUDY_RECORD_NAME)
    if search_dir is None:
        raise FileNotFoundError(missing)
    log(f"The stored copy is not in {store_dir}. Searching {search_dir}...")
    level = [search_dir]
    for _ in range(max_depth + 1):
        subfolders: list[Path] = []
        for folder in level:
            try:
                with os.scandir(folder) as entries:
                    items = list(entries)
            except OSError:
                continue
            if wanted <= {item.name for item in items if item.is_file()}:
                log(f"Found the stored copy in {folder}.")
                return folder
            subfolders.extend(
                Path(item.path) for item in items if item.is_dir() and not item.name.startswith(".")
            )
        level = sorted(subfolders)
    raise FileNotFoundError(
        f"{missing} No folder under {search_dir} holds both {STUDY_RECORD_NAME} and "
        f"{STUDY_ARCHIVE_NAME}. If they were uploaded with another Google account, add a "
        "shortcut to their folder in My Drive."
    )


def restore_study_files(store_dir: Path, local_dir: Path, log: Logger = _log) -> Path:
    """Copy the stored study archive to local disk, verify it, and extract it.

    Args:
        store_dir: Folder holding the stored archive and its record.
        local_dir: Local working folder.
        log: Receives progress messages.

    Returns:
        The corpus folder, which contains `Session1` to `Session5`.

    Raises:
        FileNotFoundError: If the stored archive or its record is missing.
        ValueError: If the copy does not match the recorded checksum or file list.
    """
    for name in (STUDY_RECORD_NAME, STUDY_ARCHIVE_NAME):
        if not (store_dir / name).is_file():
            raise FileNotFoundError(_describe_missing(store_dir / name))
    record = json.loads((store_dir / STUDY_RECORD_NAME).read_text(encoding="utf-8"))
    local_archive = local_dir / STUDY_ARCHIVE_NAME
    if not (
        local_archive.exists() and _sha256_with_retries(local_archive) == record["archive_sha256"]
    ):
        log("Copying the study archive to this runtime...")
        local_archive.unlink(missing_ok=True)
        copy_with_retries(store_dir / STUDY_ARCHIVE_NAME, local_archive, log=log)
        if _sha256_with_retries(local_archive) != record["archive_sha256"]:
            raise ValueError("The copied study archive does not match its recorded checksum.")
    corpus_dir = local_dir / "iemocap"
    with tarfile.open(local_archive) as archive:
        if set(archive.getnames()) != set(record["files"]):
            raise ValueError("The study archive's file list differs from its record.")
        archive.extractall(corpus_dir, filter="data")
    log(f"Study files verified and extracted to {corpus_dir}.")
    return corpus_dir


def check_corpus(corpus_dir: Path) -> dict[str, Any]:
    """Read the four-class utterances and compare their counts with the published counts.

    Args:
        corpus_dir: Folder containing `Session1` to `Session5`.

    Returns:
        Counts by emotion, session, and interaction type.

    Raises:
        ValueError: If the emotion counts differ from the published counts.
    """
    utterances = read_iemocap(corpus_dir, SESSIONS)
    missing_audio = [u.utterance_id for u in utterances if not (corpus_dir / u.audio_path).exists()]
    if missing_audio:
        raise ValueError(f"{len(missing_audio)} utterances have no audio file.")
    emotions = dict(sorted(Counter(u.emotion for u in utterances).items()))
    if emotions != dict(sorted(IEMOCAP_FOUR_CLASS_COUNTS.items())):
        raise ValueError(f"Emotion counts {emotions} differ from {IEMOCAP_FOUR_CLASS_COUNTS}.")
    return {
        "utterances": len(utterances),
        "emotions": emotions,
        "sessions": dict(sorted(Counter(u.session for u in utterances).items())),
        "interactions": dict(sorted(Counter(u.interaction for u in utterances).items())),
    }


def _store(options: argparse.Namespace, password: str) -> None:
    work_dir: Path = options.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    if options.source_archive is not None:
        release = options.source_archive
    else:
        release = work_dir / PurePosixPath(urllib.parse.urlsplit(options.source_url).path).name
        if not release.exists():
            _log("Downloading the release archive...")
            download_archive(options.source_url, options.username, password, release)
    _log("Computing the release archive's checksums...")
    source_checksums = compute_checksums(release)
    if options.expected_md5 is None:
        _log(
            "No reference MD5 was given. The archive is checked through its gzip checksum "
            "and the published utterance counts."
        )
    elif source_checksums["md5"] != options.expected_md5.lower():
        raise ValueError(
            f"{release.name} does not match the expected MD5. It is kept for diagnosis."
        )

    _log("Building the study archive...")
    study_archive = work_dir / STUDY_ARCHIVE_NAME
    record = build_study_archive(release, study_archive)
    record["source"] = {
        "archive": release.name,
        "bytes": release.stat().st_size,
        "checksums": source_checksums,
        "expected_md5": options.expected_md5,
        "verified_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (work_dir / STUDY_RECORD_NAME).write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = check_corpus(restore_study_files(work_dir, work_dir / "verification"))
    archive_gib = study_archive.stat().st_size / 2**30
    _log(f"Study archive: {record['file_count']} files, {archive_gib:.2f} GiB.")
    _log(f"Four-class utterances: {summary}")

    _log(f"Saving the study archive to {options.store_dir}...")
    copy_with_retries(study_archive, options.store_dir / STUDY_ARCHIVE_NAME)
    if _sha256_with_retries(options.store_dir / STUDY_ARCHIVE_NAME) != record["archive_sha256"]:
        raise ValueError("The stored study archive does not match its checksum.")
    copy_with_retries(work_dir / STUDY_RECORD_NAME, options.store_dir / STUDY_RECORD_NAME)
    _write_result(options.output_dir, {"study_archive": record["archive"], **summary})
    _log("Stored copy verified.")


def _check(options: argparse.Namespace) -> None:
    store_dir = locate_study_store(options.store_dir, options.search_dir)
    corpus_dir = restore_study_files(store_dir, options.local_dir)
    summary = check_corpus(corpus_dir)
    result = {"store_dir": str(store_dir), "corpus_dir": str(corpus_dir), **summary}
    _write_result(options.output_dir, result)
    _log(f"Four-class utterances: {summary}")


def _write_result(output_dir: Path, result: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    (output_dir / CHECK_RESULT_NAME).write_text(text, encoding="utf-8")


def main(arguments: Sequence[str] | None = None) -> None:
    """Parse command-line arguments and run the requested command."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    store = commands.add_parser("store", help="Download, reduce, verify, and store the corpus.")
    source = store.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-url", help="Download the release archive from this address.")
    source.add_argument("--source-archive", type=Path, help="Use an already downloaded archive.")
    store.add_argument("--username", help="Username for the download; required with a URL.")
    store.add_argument("--expected-md5", help="Reference MD5; required with a URL.")
    store.add_argument("--work-dir", type=Path, required=True)
    store.add_argument("--store-dir", type=Path, required=True)
    store.add_argument("--output-dir", type=Path, required=True)
    check = commands.add_parser("check", help="Restore the stored corpus and check it.")
    check.add_argument("--store-dir", type=Path, required=True)
    check.add_argument(
        "--search-dir", type=Path, help="Folder to search if the files are not in --store-dir."
    )
    check.add_argument("--local-dir", type=Path, required=True)
    check.add_argument("--output-dir", type=Path, required=True)
    options = parser.parse_args(arguments)
    downloads = options.command == "store" and options.source_url is not None
    if downloads and not (options.username and options.expected_md5):
        parser.error("--source-url requires --username and --expected-md5.")

    if options.command == "store":
        completion, failure = STORE_COMPLETION_MARKER, STORE_FAILURE_MARKER
    else:
        completion, failure = CHECK_COMPLETION_MARKER, CHECK_FAILURE_MARKER
    try:
        if options.command == "store":
            password = sys.stdin.readline().rstrip("\r\n") if downloads else ""
            stored = options.store_dir / STUDY_RECORD_NAME
            if downloads and not password and not stored.exists():
                raise DownloadError("No download password was given.")
            _store(options, password)
        else:
            _check(options)
    except Exception:
        traceback.print_exc()
        print(failure, flush=True)
        raise
    print(completion, flush=True)


if __name__ == "__main__":
    main()
