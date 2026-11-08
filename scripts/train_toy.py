
import torch
from torch.utils.data import DataLoader

from maskforge.data.tokenizer import MaskForgeTokenizer
from maskforge.data.dataset import MaskForgeDataset
from torch.optim import AdamW

from maskforge.models.denoiser import MaskForgeDenoiser
from maskforge.training.train import train


def main():
    # 1. Select device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # 2. Small toy dataset
    texts = [
        "The cat is sitting on the mat.",
        "The dog is playing in the garden.",
        "Machine learning is a branch of artificial intelligence.",
        "Deep learning uses neural networks to learn patterns.",
        "Diffusion models gradually corrupt and reconstruct data.",
        "Transformers use attention mechanisms to process sequences.",
        "Masked language models predict missing tokens from context.",
        "PyTorch is a popular framework for deep learning.",
    ]

    # 3. Load tokenizer
    tokenizer = MaskForgeTokenizer.train(
       texts,
       vocab_size=500,
    )

    tokenizer.save("data/tokenizer.json")

    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # 4. Create dataset
    dataset = MaskForgeDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=32,
    )

    print(f"Number of training examples: {len(dataset)}")

    # 5. Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        drop_last=False,
    )
    model = MaskForgeDenoiser(
       vocab_size=tokenizer.vocab_size,
       max_seq_len=32,
       pad_token_id=tokenizer.pad_token_id,
       hidden_dim=64,
       num_heads=4,
       num_layers=2,
       ff_dim=128,
       dropout=0.0,
    ).to(device)

    optimizer = AdamW(
       model.parameters(),
       lr=1e-3,
    )

    train(
       model=model,
       optimizer=optimizer,
       dataloader=dataloader,
       tokenizer=tokenizer,
       device=device,
       num_epochs=20,
       checkpoint_dir="checkpoints/toy",
    )

    # Inspect one batch
    batch = next(iter(dataloader))

    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")


if __name__ == "__main__":
    main()
