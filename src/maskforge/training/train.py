import torch 
from torch.utils.data import DataLoader
from maskforge.training.train_step import train_step 
from torch.optim import AdamW
from maskforge.models.denoiser import MaskForgeDenoiser
from maskforge.data.tokenizer import MaskForgeTokenizer
from maskforge.training.checkpoint import save_checkpoint
from pathlib import Path

def train(model: MaskForgeDenoiser, optimizer: AdamW, dataloader: DataLoader, tokenizer: MaskForgeTokenizer, device: torch.device, start_epoch: int = 0, num_epochs: int = 10, checkpoint_dir: str | Path = "checkpoints",):
    checkpoint_dir = Path(checkpoint_dir)
    for epoch in range(start_epoch, num_epochs):
        batch_loss = 0.0
        for batch in dataloader: 
            loss = train_step(model, optimizer, batch, tokenizer, device)
            batch_loss += loss 
        epoch_loss = batch_loss/len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        save_checkpoint(model, optimizer, epoch, f"checkpoints/epoch_{epoch + 1}.pt")