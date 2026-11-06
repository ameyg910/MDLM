import math 

import torch 
import torch.nn as nn 


class SinusoidalTimeEmbedding(nn.Module): 

    def __init__(self, emb_dim: int):
        super().__init__()
        self.emb_dim = emb_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.emb_dim // 2 

        frequencies = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=t.device, dtype=torch.float32,)/max(half_dim-1, 1)
        )

        angles = t.float().unsqueeze(1)*frequencies.unsqueeze(0)

        embedding = torch.cat(
            [torch.sin(angles), torch.cos(angles)], 
            dim = 1
        )

        if self.emb_dim % 2 == 1: 
            embedding = nn.functional.pad(embedding, (0, 1),)

        return embedding 

class MaskForgeDenoiser(nn.Module): 

    def __init__(self, vocab_size: int, max_seq_len: int, pad_token_id: int, hidden_dim: int=256, num_heads: int = 8, num_layers: int = 6, ff_dim: int = 1024, dropout: float = 0.1, ):
        super().__init__()
        if hidden_dim%num_heads != 0:
            raise ValueError("Hidden dim not div by num of heads")

        self.max_seq_len = max_seq_len 

        self.token_emb = nn.Embedding(vocab_size, hidden_dim, padding_idx = pad_token_id)
        self.position_embedding = nn.Embedding(
            max_seq_len,
            hidden_dim,
        )
        self.time_embedding = SinusoidalTimeEmbedding(hidden_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        t: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, seq_length = input_ids.shape

        if seq_length > self.max_seq_len:
            raise ValueError("Sequence exceeds max_seq_len.")

        if t.shape != (batch_size,):
            raise ValueError("t must have shape (batch_size,).")

        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask must have the same shape as input_ids."
            )

        token_embeddings = self.token_emb(input_ids)

        positions = torch.arange(
            seq_length,
            device=input_ids.device,
        )

        position_embeddings = self.position_embedding(positions)
        time_embeddings = self.time_embedding(t)
        time_embeddings = self.time_mlp(time_embeddings)

        hidden_states = (
            token_embeddings
            + position_embeddings.unsqueeze(0)
            + time_embeddings.unsqueeze(1)
        )
        padding_mask = attention_mask == 0
        hidden_states = self.transformer(
            hidden_states,
            src_key_padding_mask=padding_mask,
        )
        logits = self.lm_head(hidden_states)

        return logits 