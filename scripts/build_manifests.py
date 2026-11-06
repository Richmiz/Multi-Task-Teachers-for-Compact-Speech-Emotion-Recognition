"""Build corpus manifests from local dataset folders.

Manifests contain licensed labels and transcripts, so write them under `data/`, which git ignores.

    python scripts/build_manifests.py --iemocap-root data/iemocap/IEMOCAP_full_release
    python scripts/build_manifests.py --ravdess-root data/ravdess --emodb-root data/emodb
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from compact_ser.corpora import read_emodb, read_iemocap, read_ravdess, write_manifest
from compact_ser.study_design import load_study_design

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# Published four-class counts: Gao et al., Interspeech 2023, section 3.1.
EXPECTED_IEMOCAP_COUNTS = {"happy": 1636, "sad": 1084, "angry": 1103, "neutral": 1708}


def main() -> None:
    """Parse command-line arguments and write the requested manifests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iemocap-root", type=Path)
    parser.add_argument("--ravdess-root", type=Path)
    parser.add_argument("--emodb-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "data" / "manifests")
    options = parser.parse_args()

    if options.iemocap_root:
        utterances = read_iemocap(options.iemocap_root, load_study_design().sessions)
        counts = Counter(utterance.emotion for utterance in utterances)
        print(f"IEMOCAP: {len(utterances)} utterances, {dict(counts)}")
        if dict(counts) != EXPECTED_IEMOCAP_COUNTS:
            raise SystemExit(f"IEMOCAP counts differ from {EXPECTED_IEMOCAP_COUNTS}.")
        write_manifest(utterances, options.output_dir / "iemocap.jsonl")
    if options.ravdess_root:
        utterances = read_ravdess(options.ravdess_root)
        print(f"RAVDESS: {len(utterances)} utterances")
        write_manifest(utterances, options.output_dir / "ravdess.jsonl")
    if options.emodb_root:
        utterances = read_emodb(options.emodb_root)
        print(f"Emo-DB: {len(utterances)} utterances")
        write_manifest(utterances, options.output_dir / "emodb.jsonl")


if __name__ == "__main__":
    main()
