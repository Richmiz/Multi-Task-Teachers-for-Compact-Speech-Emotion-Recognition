import pytest

from compact_ser.text_processing import (
    BLANK_TOKEN,
    UNKNOWN_TOKEN,
    TranscriptTokenizer,
    edit_distance,
    find_seen_transcripts,
    normalize_transcript,
    word_error_rate,
)


def test_normalization_removes_events_and_punctuation():
    assert (
        normalize_transcript("Excuse me, [LAUGHTER] it's  <noise> fine.") == "EXCUSE ME IT'S FINE"
    )


def test_tokenizer_vocabulary_comes_from_training_text_only():
    tokenizer = TranscriptTokenizer.fit(["ab"])
    assert tokenizer.vocabulary_size == 4
    assert tokenizer.encode("az")[1] == UNKNOWN_TOKEN


def test_ctc_decoding_collapses_repeats_and_removes_blanks():
    tokenizer = TranscriptTokenizer.fit(["ab"])
    a, b = tokenizer.encode("ab")
    frames = [BLANK_TOKEN, a, a, BLANK_TOKEN, a, b, b, UNKNOWN_TOKEN, BLANK_TOKEN]
    assert tokenizer.decode_ctc(frames) == "AAB"


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [("", "", 0), ("abc", "abc", 0), ("abc", "abd", 1), ("abc", "", 3), ("kitten", "sitting", 3)],
)
def test_edit_distance(first, second, expected):
    assert edit_distance(first, second) == expected
    assert edit_distance(second, first) == expected


def test_edit_distance_limit_stops_with_a_value_above_the_limit():
    assert edit_distance("aaaaaaaa", "bbbbbbbb", limit=2) == 3
    assert edit_distance("abc", "abcdefgh", limit=2) == 3
    assert edit_distance("abc", "abd", limit=2) == 1


def test_word_error_rate_counts_edits_across_all_references():
    assert word_error_rate(["A B C D", "E"], ["A X C", "E"]) == pytest.approx(2 / 5)


def test_seen_transcripts_include_exact_and_near_matches_only():
    training = ["The quick brown fox jumps over it", "Hello there"]
    test = [
        "the quick brown fox, jumps over it!",  # exact after normalization
        "The quick brown fox jumps over its",  # one character away
        "Something else entirely",
        "Hello",
    ]
    assert find_seen_transcripts(test, training, 0.1) == [True, True, False, False]
