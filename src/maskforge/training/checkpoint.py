import torch 

from torch.optim import AdamW
from maskforge.models.denoiser import MaskForgeDenoiser
from pathlib import Path

def save_checkpoint(model: MaskForgeDenoiser, optimizer: AdamW, epoch, path: str | Path,):
      path = Path(path)
      path.parent.mkdir(parents=True, exist_ok=True)
      
      checkpoint = {
      "epoch": epoch + 1,
      "model_state_dict": model.state_dict(),
      "optimizer_state_dict": optimizer.state_dict(),
      }

      torch.save(checkpoint, path)

def load_checkpoint(model: MaskForgeDenoiser, optimizer: AdamW, path: str | Path, device=torch.device,): 
      checkpoint = torch.load(path, map_location=device, weights_only=True)
      model.load_state_dict(checkpoint["model_state_dict"])
      optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
      return checkpoint["epoch"]
