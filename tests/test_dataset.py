
import torch
from torch.utils.data import DataLoader

from maskforge.data.tokenizer import MaskForgeTokenizer
from maskforge.data.dataset import MaskForgeDataset


def create_tokenizer():
    texts = [
        "The cat sat on the mat.",
        "The dog sat on the floor.",
        "Diffusion models generate text.",
    ] * 100

    return MaskForgeTokenizer.train(
        texts,
        vocab_size=300,
    )


def test_dataset_output():

    tokenizer = create_tokenizer()

    texts = [
        "The cat sat on the mat.",
        "The dog sat on the floor.",
    ]

    dataset = MaskForgeDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=16,
    )

    assert len(dataset) == 2

    example = dataset[0]

    assert example["input_ids"].shape == (16,)
    assert example["attention_mask"].shape == (16,)

    assert example["input_ids"].dtype == torch.long
    assert example["attention_mask"].dtype == torch.long

    assert example["input_ids"][0].item() == tokenizer.bos_token_id

    assert tokenizer.eos_token_id in example["input_ids"].tolist()


def test_dataset_padding():

    tokenizer = create_tokenizer()

    dataset = MaskForgeDataset(
        texts=["cat"],
        tokenizer=tokenizer,
        max_length=16,
    )

    example = dataset[0]

    input_ids = example["input_ids"]
    attention_mask = example["attention_mask"]

    # All positions after EOS should be padding.
    eos_position = (
        input_ids == tokenizer.eos_token_id
    ).nonzero(as_tuple=True)[0].item()

    assert torch.all(
        input_ids[eos_position + 1:] == tokenizer.pad_token_id
    )

    assert torch.all(attention_mask[:eos_position + 1] == 1)
    assert torch.all(attention_mask[eos_position + 1:] == 0)


def test_long_document_chunking():

    tokenizer = create_tokenizer()

    dataset = MaskForgeDataset(
        texts=["cat " * 100],
        tokenizer=tokenizer,
        max_length=16,
    )

    assert len(dataset) > 1

    for example in dataset:
        assert example["input_ids"].shape == (16,)
        assert example["input_ids"][0].item() == tokenizer.bos_token_id


def test_dataloader_batching():

    tokenizer = create_tokenizer()

    dataset = MaskForgeDataset(
        texts=[
            "The cat sat on the mat.",
            "The dog sat on the floor.",
            "Diffusion models generate text.",
            "Transformers process sequences.",
        ],
        tokenizer=tokenizer,
        max_length=16,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
    )

    batch = next(iter(loader))

    assert batch["input_ids"].shape == (2, 16)
    assert batch["attention_mask"].shape == (2, 16)
