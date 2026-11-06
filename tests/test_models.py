import pytest
import torch
from transformers import Wav2Vec2Model

from compact_ser.losses import (
    distillation_loss,
    hidden_state_loss,
    multitask_loss,
    transcript_loss,
)
from compact_ser.models import (
    SpeechEmotionModel,
    build_student,
    compute_frame_lengths,
    select_evenly_spaced_layers,
    small_encoder_config,
)

NUM_EMOTIONS = 4


@pytest.fixture(scope="module")
def encoder():
    torch.manual_seed(0)
    return Wav2Vec2Model(small_encoder_config())


@pytest.fixture
def batch():
    torch.manual_seed(1)
    waveforms = 0.1 * torch.randn(2, 8000)
    attention_mask = torch.ones_like(waveforms, dtype=torch.long)
    attention_mask[1, 4000:] = 0
    return waveforms * attention_mask, attention_mask


@pytest.mark.parametrize(
    ("depth", "expected"),
    [(6, (2, 4, 6, 8, 10, 12)), (4, (3, 6, 9, 12))],
)
def test_layer_selection_matches_the_study_rule(depth, expected):
    assert select_evenly_spaced_layers(12, depth) == expected


def test_layers_that_cannot_be_spaced_evenly_are_rejected():
    with pytest.raises(ValueError, match="evenly"):
        select_evenly_spaced_layers(12, 5)


def test_frame_lengths_match_the_encoder_output(encoder, batch):
    waveforms, attention_mask = batch
    encoder.eval()
    with torch.no_grad():
        frames = encoder(waveforms, attention_mask=attention_mask).last_hidden_state
    lengths = compute_frame_lengths(attention_mask.sum(dim=-1), encoder.config)
    assert lengths[0] == frames.size(1)
    assert lengths[1] < lengths[0]


def predictions_with_and_without_padding(feature_norm):
    torch.manual_seed(2)
    encoder = Wav2Vec2Model(small_encoder_config(feature_norm=feature_norm))
    model = SpeechEmotionModel(encoder, NUM_EMOTIONS).eval()
    audio = 0.1 * torch.randn(1, 6000)
    padded = torch.cat([audio, torch.zeros(1, 2000)], dim=1)
    mask = torch.cat([torch.ones(1, 6000), torch.zeros(1, 2000)], dim=1).long()
    with torch.no_grad():
        unpadded_logits = model(audio, torch.ones_like(audio, dtype=torch.long)).emotion_logits
        padded_logits = model(padded, mask).emotion_logits
    return unpadded_logits, padded_logits


def test_pooling_ignores_padding_when_the_frontend_does_not_use_it():
    unpadded_logits, padded_logits = predictions_with_and_without_padding("layer")
    assert torch.allclose(unpadded_logits, padded_logits, atol=1e-4)


def test_group_normalization_makes_predictions_depend_on_padding():
    unpadded_logits, padded_logits = predictions_with_and_without_padding("group")
    assert not torch.allclose(unpadded_logits, padded_logits, atol=1e-4)


def test_student_copies_selected_teacher_layers_and_leaves_the_teacher_unchanged(encoder):
    student = build_student(encoder, depth=6, num_emotions=NUM_EMOTIONS)
    assert len(student.encoder.encoder.layers) == 6
    assert len(encoder.encoder.layers) == 12
    for student_layer, teacher_number in zip(
        student.encoder.encoder.layers, (2, 4, 6, 8, 10, 12), strict=True
    ):
        source = encoder.encoder.layers[teacher_number - 1].state_dict()
        for name, value in student_layer.state_dict().items():
            assert torch.equal(value, source[name])
    assert student.encoder.encoder.layers[0] is not encoder.encoder.layers[1]


def test_student_has_only_an_emotion_head(encoder, batch):
    student = build_student(encoder, depth=4, num_emotions=NUM_EMOTIONS)
    output = student(*batch)
    assert output.emotion_logits.shape == (2, NUM_EMOTIONS)
    assert output.transcript_logits is None
    assert output.speaker_logits is None


def test_all_multitask_losses_reach_the_encoder(encoder, batch):
    torch.manual_seed(3)
    model = SpeechEmotionModel(encoder, NUM_EMOTIONS, vocabulary_size=8, num_speakers=6).train()
    model.zero_grad()
    output = model(*batch)
    loss = multitask_loss(
        output,
        emotion_labels=torch.tensor([0, 1]),
        transcript_targets=torch.tensor([[1, 2, 3], [4, 5, 0]]),
        transcript_lengths=torch.tensor([3, 2]),
        speaker_labels=torch.tensor([0, 5]),
        transcript_weight=0.5,
        speaker_weight=0.5,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.speaker_head.weight.grad is not None
    assert model.transcript_head.weight.grad is not None
    assert encoder.encoder.layers[-1].final_layer_norm.weight.grad is not None
    model.zero_grad()


def test_inactive_auxiliary_losses_do_not_need_labels(encoder, batch):
    model = SpeechEmotionModel(encoder, NUM_EMOTIONS)
    loss = multitask_loss(model(*batch), emotion_labels=torch.tensor([0, 1]))
    assert torch.isfinite(loss)


def test_hidden_state_loss_ignores_padded_frames():
    student = (torch.zeros(1, 3, 2), torch.zeros(1, 3, 2))
    teacher = [torch.zeros(1, 3, 2), torch.zeros(1, 3, 2)]
    teacher[1][0, 2] = 5.0
    frame_lengths = torch.tensor([2])
    assert hidden_state_loss(student, teacher, [1], frame_lengths) == 0.0
    assert hidden_state_loss(student, teacher, [1], torch.tensor([3])) > 0.0


def test_distillation_updates_only_the_student(encoder, batch):
    torch.manual_seed(4)
    teacher = SpeechEmotionModel(encoder, NUM_EMOTIONS).eval()
    student = build_student(encoder, depth=6, num_emotions=NUM_EMOTIONS).train()
    teacher.zero_grad()
    with torch.no_grad():
        teacher_output = teacher(*batch, output_hidden_states=True)
    student_output = student(*batch, output_hidden_states=True)
    loss = distillation_loss(
        student_output,
        teacher_output,
        (2, 4, 6, 8, 10, 12),
        torch.tensor([0, 1]),
        temperature=2.0,
        logit_weight=1.0,
        hidden_state_weight=1.0,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert student.emotion_head.weight.grad is not None
    assert teacher.emotion_head.weight.grad is None


def test_transcript_loss_accepts_an_empty_transcript(encoder, batch):
    model = SpeechEmotionModel(encoder, NUM_EMOTIONS, vocabulary_size=8)
    output = model(*batch)
    targets = torch.tensor([[1, 2], [0, 0]])
    loss = transcript_loss(output, targets, torch.tensor([2, 0]))
    assert torch.isfinite(loss)
