from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Hyperparameters for the GPT-style transformer."""

    # Architecture
    num_layers: int = 8
    d_model: int = 512
    num_heads: int = 8
    dff: int = 2048
    vocab_size: int = 100_277  # tiktoken cl100k_base
    max_seq_len: int = 1024
    dropout: float = 0.1

    def __post_init__(self) -> None:
        assert (
            self.d_model % self.num_heads == 0
        ), f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
