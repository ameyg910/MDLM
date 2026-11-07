import torch 

import torch.nn.functional as F

def mdlm_loss(logits: torch.Tensor, input_ids: torch.Tensor, mask_positions: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3: 
        raise ValueError("logits shape must be (B, L, V).")

    if input_ids.shape != logits.shape[:2]: 
        raise ValueError("input_ids must have shape (B, L).")

    if mask_positions.shape != input_ids.shape:
        raise ValueError("mask_positions must have shape (B, L).")

    if t.shape != (input_ids.shape[0],):
        raise ValueError("t must have shape (B,).")

    mask_positions = mask_positions.bool()

    if not mask_positions.any():
        return logits.sum() * 0.0

    batch_size, seq_length, vocab_size = logits.shape

    per_token_loss = F.cross_entropy(logits.reshape(-1, vocab_size), input_ids.reshape(-1), reduction="none",).reshape(batch_size, seq_length)
    weights = (1.0/t).unsqueeze(1)

    weighted_loss = (
    per_token_loss
    * weights
    * mask_positions.float()
    )
    per_sequence_loss = weighted_loss.sum(dim=1)
    loss = per_sequence_loss.mean()
    return loss 