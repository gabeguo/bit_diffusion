import torch

class SharedTokenDecoder(torch.nn.Module):
    """Shared per-token MLP: each token embedding predicts one token ID."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 128,
        token_seq_len: int = 64,
        token_emb_dim: int = 64,
    ):
        super().__init__()
        self.token_seq_len = int(token_seq_len)
        self.token_emb_dim = int(token_emb_dim)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.token_emb_dim, hidden_dim, bias=True),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim, bias=True),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, vocab_size, bias=True),
        )

    def forward(self, token_flat: torch.Tensor) -> torch.Tensor:
        bsz = token_flat.shape[0]
        x = token_flat.view(bsz, self.token_seq_len, self.token_emb_dim)
        x = torch.nn.functional.normalize(x, p=2, dim=-1) # undo the dataest scaling
        return self.net(x)
