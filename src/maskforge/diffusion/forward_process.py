import torch 

def forward_corrupt(
        input_ids: torch.Tensor, 
        t: torch.Tensor, 
        mask_token_id: int, 
        pad_token_id: int, 
        bos_token_id: int, 
        eos_token_id: int, 
):

    if input_ids.ndim != 2: 
        raise  ValueError("input_ids should have shape (batch_size, sequence_length). ")

    batch_size = input_ids.shape[0]

    if t.ndim != 1 or t.shape[0] != batch_size: 
        raise ValueError("t should have shape (batch_size,).")

    t = t.to(device=input_ids.device, dtype=torch.float32)

    random_values = torch.rand(
        input_ids.shape, 
        device=input_ids.device, 
    )

    t = t.unsqueeze(1) 

    random_mask = random_values < t

    special_positions = (
        (input_ids == pad_token_id)
        | (input_ids == bos_token_id)
        | (input_ids == eos_token_id)
    )

    mask_positions = random_mask & ~special_positions
    noisy_input_ids = input_ids.clone()

    noisy_input_ids[mask_positions] = mask_token_id

    return noisy_input_ids, mask_positions