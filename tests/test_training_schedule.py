import itertools
from dataclasses import replace

import pytest
import torch
from torch.nn import functional
from transformers import Wav2Vec2Model

from compact_ser.models import SpeechEmotionModel, small_encoder_config
from compact_ser.study_design import load_study_design
from compact_ser.study_runner import LEDGER_FILE_NAME, check_study_settings
from compact_ser.training import learning_rate_factor, train_stage

SETTINGS = load_study_design().training


@pytest.mark.parametrize(
    ("update", "expected"),
    [(1, 1.0), (10, 1.0), (11, 1 / 9), (19, 1.0), (55, 1.0), (56, 44 / 45), (100, 0.0)],
)
def test_schedule_trains_heads_then_warms_up_holds_and_decays(update, expected):
    # 100 updates: 10 heads only, then 9 warm-up, 36 hold, and 45 decay updates.
    assert learning_rate_factor(update, 100, SETTINGS) == pytest.approx(expected)


@pytest.fixture
def model():
    torch.manual_seed(0)
    encoder = Wav2Vec2Model(small_encoder_config())
    encoder.freeze_feature_encoder()
    return SpeechEmotionModel(encoder, 4)


def test_heads_train_alone_before_the_encoder(model):
    torch.manual_seed(1)
    waveforms = 0.1 * torch.randn(2, 8000)
    mask = torch.ones_like(waveforms, dtype=torch.long)
    labels = torch.tensor([0, 1])

    def compute_loss(_):
        output = model(waveforms, mask)
        return {"emotion": functional.cross_entropy(output.emotion_logits, labels)}, {}

    start = {name: value.clone() for name, value in model.state_dict().items()}
    seen = []

    def on_encoder_start():
        state = model.state_dict()
        encoder_same = all(
            torch.equal(state[k], start[k]) for k in start if k.startswith("encoder.")
        )
        head_moved = any(not torch.equal(state[k], start[k]) for k in start if "emotion_head" in k)
        seen.append((encoder_same, head_moved))

    settings = replace(
        SETTINGS, head_only_share=0.5, gradient_log_interval=1, validation_interval=100
    )
    outcome = train_stage(
        model, compute_loss, itertools.repeat(None), 6, settings, on_encoder_start=on_encoder_start
    )
    assert seen == [(True, True)]
    state = model.state_dict()
    layers = [name for name in start if name.startswith("encoder.encoder.")]
    assert any(not torch.equal(state[name], start[name]) for name in layers)
    assert [entry["update"] for entry in outcome.gradient_history] == [4, 5, 6]
    assert not any(p.requires_grad for p in model.encoder.feature_extractor.parameters())
    assert all(p.requires_grad for p in model.encoder.encoder.parameters())


def test_study_folder_keeps_one_set_of_settings(tmp_path):
    design = load_study_design()
    check_study_settings(tmp_path, design)
    check_study_settings(tmp_path, design)
    changed = replace(design, training=replace(design.training, learning_rate=1e-5))
    with pytest.raises(ValueError, match=r"other settings \(training\)"):
        check_study_settings(tmp_path, changed)


def test_runs_without_a_settings_record_are_refused(tmp_path):
    (tmp_path / LEDGER_FILE_NAME).write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="without a record of their settings"):
        check_study_settings(tmp_path, load_study_design())
