from collections import Counter

import pytest

from compact_ser.corpora import (
    Partition,
    SealedPartitionError,
    read_emodb,
    read_iemocap,
    read_manifest,
    read_ravdess,
    select_partition,
    write_manifest,
)
from compact_ser.study_design import session_folds

SESSIONS = (1, 2, 3, 4, 5)


@pytest.fixture(scope="module")
def utterances(iemocap_root):
    return read_iemocap(iemocap_root, SESSIONS)


def test_iemocap_labels_are_mapped_and_other_labels_are_excluded(utterances):
    # Each dialog has 10 utterances, and one of them has an excluded label.
    assert len(utterances) == 5 * 2 * 9
    counts = Counter(utterance.emotion for utterance in utterances)
    assert counts == {"neutral": 20, "happy": 30, "sad": 20, "angry": 20}
    assert Counter(utterance.source_label for utterance in utterances)["exc"] == 10


def test_iemocap_speakers_and_interactions_are_read_from_identifiers(utterances):
    assert {utterance.speaker for utterance in utterances if utterance.session == 1} == {
        "Ses01_F",
        "Ses01_M",
    }
    interactions = {utterance.utterance_id: utterance.interaction for utterance in utterances}
    assert interactions["Ses01F_impro01_F000"] == "improvised"
    assert interactions["Ses01M_script01_1_F000"] == "scripted"


def test_iemocap_audio_paths_exist(utterances, iemocap_root):
    assert all((iemocap_root / utterance.audio_path).exists() for utterance in utterances)


def test_missing_iemocap_session_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_iemocap(tmp_path, [1])


def test_manifest_round_trip_keeps_every_field(utterances, tmp_path):
    path = tmp_path / "manifests" / "iemocap.jsonl"
    write_manifest(utterances, path)
    assert read_manifest(path) == utterances


def test_test_partition_is_sealed_by_default(utterances):
    fold = session_folds(SESSIONS)[0]
    with pytest.raises(SealedPartitionError):
        select_partition(utterances, fold, Partition.TEST)


def test_partitions_are_disjoint_and_cover_the_corpus(utterances):
    fold = session_folds(SESSIONS)[0]
    training = select_partition(utterances, fold, Partition.TRAINING)
    validation = select_partition(utterances, fold, Partition.VALIDATION)
    test = select_partition(utterances, fold, Partition.TEST, unseal_test=True)
    assert {utterance.session for utterance in training} == {3, 4, 5}
    assert {utterance.session for utterance in validation} == {2}
    assert {utterance.session for utterance in test} == {1}
    assert len(training) + len(validation) + len(test) == len(utterances)
    speakers = [{utterance.speaker for utterance in part} for part in (training, validation, test)]
    assert not (speakers[0] & speakers[1] or speakers[0] & speakers[2] or speakers[1] & speakers[2])


def test_ravdess_keeps_only_included_audio_only_speech(tmp_path):
    names = [
        "03-01-01-01-01-01-12.wav",  # neutral, included
        "03-01-05-02-02-02-12.wav",  # angry, included
        "03-01-02-01-01-01-12.wav",  # calm, excluded
        "03-02-05-01-01-01-12.wav",  # song, excluded
    ]
    folder = tmp_path / "Actor_12"
    folder.mkdir()
    for name in names:
        (folder / name).touch()
    utterances = read_ravdess(tmp_path)
    assert [utterance.emotion for utterance in utterances] == ["neutral", "angry"]
    assert utterances[0].speaker == "Actor_12"
    assert utterances[1].transcript == "Dogs are sitting by the door"


def test_emodb_keeps_only_included_emotions(tmp_path):
    for name in ("03a01Wa.wav", "03a01Fa.wav", "10b02Ta.wav", "10b02Na.wav", "03a01La.wav"):
        (tmp_path / name).touch()
    utterances = read_emodb(tmp_path)
    assert Counter(utterance.emotion for utterance in utterances) == {
        "angry": 1,
        "happy": 1,
        "sad": 1,
        "neutral": 1,
    }
    assert {utterance.speaker for utterance in utterances} == {"03", "10"}
