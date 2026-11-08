import torch 
from torch.optim import AdamW
from maskforge.models.denoiser import MaskForgeDenoiser
from maskforge.data.tokenizer import MaskForgeTokenizer
from maskforge.diffusion.noise_schedule import sample_timesteps
from maskforge.diffusion.forward_process import forward_corrupt 
from maskforge.diffusion.loss import mdlm_loss

def train_step(model: MaskForgeDenoiser, optimizer: AdamW, batch: dict, tokenizer: MaskForgeTokenizer, device: torch.device):
       model.train()
       input_ids = batch["input_ids"]
       attention_mask = batch["attention_mask"]
       input_ids = input_ids.to(device)
       attention_mask = attention_mask.to(device)
       batch_size = input_ids.shape[0]
       timesteps = sample_timesteps(batch_size, device)
       
       noisy_input_ids, mask_positions = forward_corrupt(input_ids, t=timesteps, mask_token_id = tokenizer.mask_token_id, pad_token_id = tokenizer.pad_token_id, eos_token_id = tokenizer.eos_token_id, bos_token_id = tokenizer.bos_token_id)
       logits = model(input_ids=noisy_input_ids, t=timesteps, attention_mask=attention_mask,)
       loss = mdlm_loss(logits=logits, input_ids=input_ids, mask_positions=mask_positions, t=timesteps,)
       optimizer.zero_grad(set_to_none=True)
       loss.backward()
       optimizer.step()
       return loss.item()