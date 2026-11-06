import torch 

def sample_timesteps(
        batch_size: int, 
        device: torch.device, 
) -> torch.Tensor:


    return torch.rand(batch_size, device=device)

def mask_probability(t: torch.Tensor) -> torch.Tensor: 

    return t 