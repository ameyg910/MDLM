from maskforge.diffusion.noise_schedule import sample_timesteps
from maskforge.diffusion.forward_process import forward_corrupt

import torch 

def test_sample_timesteps():
    t = sample_timesteps(
        batch_size=4,
        device=torch.device("cpu"),
    )

    assert t.shape == (4,)
    assert torch.all(t >= 0)
    assert torch.all(t < 1)


def test_forward_corrupt_at_zero():
    input_ids = torch.tensor([
        [101, 10, 20, 102, 0],
        [101, 30, 40, 102, 0],
    ])

    t = torch.tensor([0.0, 0.0])

    noisy, mask_positions = forward_corrupt(
        input_ids=input_ids,
        t=t,
        mask_token_id=999,
        pad_token_id=0,
        bos_token_id=101,
        eos_token_id=102,
    )

    assert torch.equal(noisy, input_ids)
    assert not mask_positions.any()

def test_forward_corrupt_at_one():
    input_ids = torch.tensor([
        [101, 10, 20, 102, 0],
        [101, 30, 40, 102, 0],
    ])

    t = torch.tensor([1.0, 1.0])

    noisy, mask_positions = forward_corrupt(
        input_ids=input_ids,
        t=t,
        mask_token_id=999,
        pad_token_id=0,
        bos_token_id=101,
        eos_token_id=102,
    )

    expected = torch.tensor([
        [101, 999, 999, 102, 0],
        [101, 999, 999, 102, 0],
    ])

    assert torch.equal(noisy, expected)

def test_original_input_is_unchanged():
    input_ids = torch.tensor([
        [101, 10, 20, 102, 0],
    ])

    original = input_ids.clone()

    forward_corrupt(
        input_ids=input_ids,
        t=torch.tensor([1.0]),
        mask_token_id=999,
        pad_token_id=0,
        bos_token_id=101,
        eos_token_id=102,
    )

    assert torch.equal(input_ids, original)
