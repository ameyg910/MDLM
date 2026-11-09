import torch

from maskforge.data.tokenizer import MaskForgeTokenizer
from maskforge.models.denoiser import MaskForgeDenoiser
from maskforge.diffusion.sampler import generate


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tokenizer = MaskForgeTokenizer.load("data/tokenizer.json")

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

    checkpoint = torch.load(
        "checkpoints/epoch_20.pt",
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    text = generate(
        model=model,
        tokenizer=tokenizer,
        max_length=32,
        num_steps=10,
        device=device,
    )

    print("Generated text:", text)


if __name__ == "__main__":
    main()