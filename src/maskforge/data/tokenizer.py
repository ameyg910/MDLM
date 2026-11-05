from pathlib import Path 

from tokenizers import Tokenizer, AddedToken
from tokenizers import models, trainers, pre_tokenizers, decoders 

class MaskForgeTokenizer: 

    SPECIAL_TOKENS = [
        "[PAD]", 
        "[UNK]",
        "[BOS]",
        "[EOS]",
        "[MASK]",
    ]

    def __init__(self, tokenizer:Tokenizer):
        self.tokenizer = tokenizer 

        self.pad_token_id = tokenizer.token_to_id("[PAD]")
        self.unk_token_id = tokenizer.token_to_id("[UNK]")
        self.bos_token_id = tokenizer.token_to_id("[BOS]")
        self.eos_token_id = tokenizer.token_to_id("[EOS]")
        self.mask_token_id = tokenizer.token_to_id("[MASK]")

        if any(
            token_id is None
            for token_id in [
                self.pad_token_id,
                self.unk_token_id,
                self.bos_token_id,
                self.eos_token_id,
                self.mask_token_id,
            ]  
        ):
            raise ValueError("Tokenizer is Missing required special tokens.")

    @classmethod
    def train(
        cls, 
        texts, 
        vocab_size: int = 16000,
    ):

        tokenizer = Tokenizer(
            models.BPE(unk_token="[UNK]")
        )

        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space = False 
        )

        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )

        tokenizer.train_from_iterator(
            texts,
            trainer=trainer,
        )

        for token in cls.SPECIAL_TOKENS:
            tokenizer.add_special_tokens([
               AddedToken(token, special=True)
            ])

        return cls(tokenizer)

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def encode(
            self,
        text: str,
        add_special_tokens: bool = True,
    ) -> list[int]:

        encoding = self.tokenizer.encode(text)

        token_ids = encoding.ids

        if add_special_tokens:
            token_ids = (
                [self.bos_token_id]
                + token_ids
                + [self.eos_token_id]
            )

        return token_ids
    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:

        if skip_special_tokens:
            special_ids = {
                self.tokenizer.token_to_id(token)
                for token in self.SPECIAL_TOKENS
            }

            token_ids = [
                token_id
                for token_id in token_ids
                if token_id not in special_ids
            ]

        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
        )

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.tokenizer.save(str(path))

    @classmethod
    def load(cls, path: str | Path):
        tokenizer = Tokenizer.from_file(str(path))
        return cls(tokenizer)