import numpy as np
import pytest
import soundfile
import torch

from compact_ser.audio_data import (
    SAMPLE_RATE,
    UNKNOWN_SPEAKER,
    EncodedUtterance,
    collate_batch,
    compute_class_weights,
    encode_utterances,
    iterate_batches,
    normalize_waveform,
    read_waveform,
)
from compact_ser.corpora import read_iemocap
from compact_ser.metrics import (
    compute_confusion_matrix,
    expected_calibration_error,
    macro_f1_score,
    negative_log_likelihood,
    unweighted_average_recall,
)
from compact_ser.text_processing import TranscriptTokenizer

CPU = torch.device("cpu")


def make_item(samples: int, emotion: int = 0, tokens=(1, 2)) -> EncodedUtterance:
    return EncodedUtterance(np.ones(samples, dtype=np.float32), emotion, 0, tokens, "AB")


def test_audio_is_resampled_to_16_khz_mono(tmp_path):
    path = tmp_path / "stereo.wav"
    soundfile.write(path, np.zeros((48_000, 2), dtype=np.float32), 48_000)
    waveform = read_waveform(path)
    assert waveform.ndim == 1
    assert waveform.size == SAMPLE_RATE


def test_normalization_gives_zero_mean_and_unit_variance():
    waveform = normalize_waveform(np.random.default_rng(0).normal(3.0, 2.0, 16_000))
    assert waveform.mean() == pytest.approx(0.0, abs=1e-6)
    assert waveform.std() == pytest.approx(1.0, abs=1e-3)


def test_speakers_outside_training_are_marked_unknown(iemocap_root):
    utterances = read_iemocap(iemocap_root, [1])[:2]
    tokenizer = TranscriptTokenizer.fit(utterance.transcript for utterance in utterances)
    speakers = {utterances[0].speaker: 0}
    encoded = encode_utterances(
        utterances, iemocap_root, ["angry", "happy", "sad", "neutral"], tokenizer, speakers
    )
    assert encoded[0].speaker == 0
    assert encoded[0].tokens
    other = encode_utterances(
        utterances, iemocap_root, ["angry", "happy", "sad", "neutral"], tokenizer, {}
    )
    assert other[0].speaker == UNKNOWN_SPEAKER


def test_batches_are_padded_with_masks_and_token_lengths():
    batch = collate_batch([make_item(4), make_item(2, tokens=(1,))], CPU)
    assert batch.waveforms.tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]
    assert batch.attention_mask.sum(dim=1).tolist() == [4, 2]
    assert batch.transcript_targets.tolist() == [[1, 2], [1, 0]]
    assert batch.transcript_lengths.tolist() == [2, 1]


def test_batches_without_tokens_are_supported():
    batch = collate_batch([make_item(2, tokens=())], CPU)
    assert batch.transcript_lengths.tolist() == [0]


def test_batch_order_depends_only_on_the_seed_and_covers_each_item_once_per_pass():
    items = [make_item(2, emotion=index) for index in range(6)]
    first = iterate_batches(items, 2, seed=5, device=CPU)
    second = iterate_batches(items, 2, seed=5, device=CPU)
    labels = [next(first).emotion_labels.tolist() for _ in range(3)]
    assert labels == [next(second).emotion_labels.tolist() for _ in range(3)]
    assert sorted(sum(labels, [])) == list(range(6))


def test_class_weights_favor_rare_classes():
    items = [make_item(1, emotion=0)] * 3 + [make_item(1, emotion=1)]
    weights = compute_class_weights(items, 2)
    assert weights.tolist() == pytest.approx([4 / 6, 2.0])


def test_uar_averages_recall_across_classes():
    labels = np.array([0, 0, 0, 1])
    predictions = np.array([0, 0, 1, 1])
    assert compute_confusion_matrix(labels, predictions, 2).tolist() == [[2, 1], [0, 1]]
    assert unweighted_average_recall(labels, predictions, 2) == pytest.approx((2 / 3 + 1) / 2)


def test_macro_f1_ignores_classes_missing_from_the_labels():
    labels = np.array([0, 0])
    predictions = np.array([0, 1])
    assert macro_f1_score(labels, predictions, 2) == pytest.approx(2 / 3)


def test_calibration_metrics():
    probabilities = np.array([[0.9, 0.1], [0.9, 0.1]])
    labels = np.array([0, 1])
    assert negative_log_likelihood(probabilities, labels) == pytest.approx(
        -(np.log(0.9) + np.log(0.1)) / 2
    )
    assert expected_calibration_error(probabilities, labels) == pytest.approx(0.4)
