import torch 

@torch.no_grad()
def reverse_step(model, input_ids: torch.Tensor, t: float, s: float, special_token_ids: list[int], mask_token_id: int, attention_mask: torch.Tensor,):
    batch_size = input_ids.shape[0]
    device = input_ids.device

    timesteps = torch.full((batch_size,), t, device=device,)
    logits = model(input_ids, t=timesteps, attention_mask=attention_mask)
    logits = logits.clone()
    logits[..., special_token_ids] = float("-inf")
    predicted_tokens = logits.argmax(dim=-1)
    mask_positions = input_ids == mask_token_id

    reveal_prob = (t-s)/t
    random_values = torch.rand(input_ids.shape, device=device,)
    reveal_positions = (mask_positions & (random_values < reveal_prob))

    updated_input_ids = input_ids.clone()
    updated_input_ids[reveal_positions] = predicted_tokens[reveal_positions]

    return updated_input_ids

@torch.no_grad()
def generate(model, tokenizer, max_length: int, num_steps: int, device: torch.device):
    model.eval()
    input_ids = torch.full((1, max_length), tokenizer.mask_token_id, dtype=torch.long, device=device,)
    input_ids[0, 0] = tokenizer.bos_token_id
    input_ids[0, -1] = tokenizer.eos_token_id
    attention_mask = torch.ones(
        (1, max_length),
        dtype=torch.long,
        device=device,
    )
    special_token_ids = [
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.mask_token_id,
    ]
    timesteps = torch.linspace(
        1.0, 0.0, num_steps + 1
    )
    for i in range(num_steps):
        t = timesteps[i].item()
        s = timesteps[i + 1].item()

        input_ids = reverse_step(
            model=model,
            input_ids=input_ids,
            t=t,
            s=s,
            mask_token_id=tokenizer.mask_token_id,
            attention_mask=attention_mask,
            special_token_ids=special_token_ids,
        )
    print([
    (token_id, tokenizer.tokenizer.id_to_token(token_id))
    for token_id in input_ids[0].tolist()
    ])
    return tokenizer.decode(
        input_ids[0].tolist(),
        skip_special_tokens=True,
    )