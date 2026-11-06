
import torch

from maskforge.models.denoiser import MaskForgeDenoiser


def create_model():
    return MaskForgeDenoiser(
        vocab_size=1000,
        max_seq_len=16,
        pad_token_id=0,
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        ff_dim=128,
        dropout=0.0,
    )


def test_output_shape():
    model = create_model()

    input_ids = torch.randint(1, 1000, (4, 16))
    t = torch.rand(4)
    attention_mask = torch.ones(4, 16, dtype=torch.long)

    logits = model(input_ids, t, attention_mask)

    assert logits.shape == (4, 16, 1000)


def test_forward_and_backward():
    model = create_model()

    input_ids = torch.randint(1, 1000, (2, 16))
    t = torch.rand(2)
    attention_mask = torch.ones(2, 16, dtype=torch.long)

    logits = model(input_ids, t, attention_mask)

    loss = logits.mean()
    loss.backward()

    assert model.token_emb.weight.grad is not None
    assert model.time_mlp[0].weight.grad is not None
    assert model.lm_head.weight.grad is not None


def test_padding_mask_shape():
    model = create_model()

    input_ids = torch.randint(1, 1000, (2, 16))
    attention_mask = torch.ones(2, 16, dtype=torch.long)

    attention_mask[:, -4:] = 0
    input_ids[:, -4:] = 0

    t = torch.rand(2)

    logits = model(input_ids, t, attention_mask)

    assert logits.shape == (2, 16, 1000)
    assert torch.isfinite(logits).all()
