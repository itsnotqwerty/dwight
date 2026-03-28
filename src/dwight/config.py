from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Hyperparameters for the GPT-style transformer."""

    # Architecture
    num_layers: int = 10
    d_model: int = 768
    num_heads: int = 12
    dff: int = 3072
    vocab_size: int = 100_277  # tiktoken cl100k_base
    max_seq_len: int = 1024
    dropout: float = 0.1

    # Multi-Head Latent Attention (MLA, DeepSeek-style compressed KV)
    use_mla: bool = True
    kv_latent_dim: int = 512  # c_KV bottleneck dimension
    q_latent_dim: int = 1536  # c_Q bottleneck dimension
    qk_rope_dim: int = 64  # per-head rope dimension appended to K and Q

    # Mixture-of-Experts FFN
    use_moe: bool = True
    num_experts: int = 8  # total routed experts
    num_active_experts: int = 2  # top-K active experts per token
    num_shared_experts: int = 1  # always-active shared experts
    expert_hidden_dim: int = 512  # SwiGLU hidden dim inside each expert
    moe_aux_loss_coeff: float = 0.01  # weight on load-balance aux loss

    def __post_init__(self) -> None:
        assert (
            self.d_model % self.num_heads == 0
        ), f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
        if self.use_mla:
            assert self.qk_rope_dim % 2 == 0, "qk_rope_dim must be even for RoPE"
