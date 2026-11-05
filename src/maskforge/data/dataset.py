
import torch
from torch.utils.data import Dataset


class MaskForgeDataset(Dataset):

    def __init__(
        self,
        texts: list[str],
        tokenizer,
        max_length: int = 128,
    ):
        if max_length < 2:
            raise ValueError("max_length must be at least 2.")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        # One position for EOS, one for BOS.
        chunk_size = max_length - 2

        for text in texts:

            token_ids = tokenizer.encode(text)

            # Remove BOS and EOS.
            content_ids = token_ids[1:-1]

            chunks = [
                content_ids[i:i + chunk_size]
                for i in range(0, len(content_ids), chunk_size)
            ]

            if not chunks:
                chunks = [[]]

            for chunk in chunks:

                sequence = (
                    [tokenizer.bos_token_id]
                    + chunk
                    + [tokenizer.eos_token_id]
                )

                attention_mask = [1] * len(sequence)

                padding_length = max_length - len(sequence)

                sequence += (
                    [tokenizer.pad_token_id] * padding_length
                )

                attention_mask += [0] * padding_length

                self.examples.append(
                    {
                        "input_ids": sequence,
                        "attention_mask": attention_mask,
                    }
                )


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):

        example = self.examples[index]

        return {
            "input_ids": torch.tensor(
                example["input_ids"],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                example["attention_mask"],
                dtype=torch.long,
            ),
        }
