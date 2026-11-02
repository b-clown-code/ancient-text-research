"""Small decoder-only GQA transformer for Ancient Model Gate 2."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class ModelConfig:
    vocab_size: int = 32768
    d_model: int = 576
    n_layers: int = 30
    n_heads: int = 9
    n_kv_heads: int = 3
    ffn_dim: int = 1728
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    tie_embeddings: bool = True
    activation_checkpointing: bool = True

    def validate(self):
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.d_model // self.n_heads % 2:
            raise ValueError("RoPE head dimension must be even")


def parameter_count(config: ModelConfig) -> int:
    """Exact count for this bias-free architecture, including final norm."""
    c = config
    head_dim = c.d_model // c.n_heads
    embedding = c.vocab_size * c.d_model
    if not c.tie_embeddings:
        embedding += c.vocab_size * c.d_model
    attention = c.d_model * c.d_model + 2 * (c.n_kv_heads * head_dim) * c.d_model + c.d_model * c.d_model
    feed_forward = 3 * c.d_model * c.ffn_dim
    norms = 2 * c.d_model
    return embedding + c.n_layers * (attention + feed_forward + norms) + c.d_model


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * scale).to(x.dtype) * self.weight


def apply_rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    # x: [batch, heads, sequence, head_dim]
    dim = x.shape[-1]
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim))
    angles = positions.to(torch.float32)[:, None] * inv_freq[None, :]
    cos, sin = angles.cos()[None, None, :, :], angles.sin()[None, None, :, :]
    pair = x.float().reshape(*x.shape[:-1], dim // 2, 2)
    even, odd = pair[..., 0], pair[..., 1]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2).to(x.dtype)


class GQAAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.rope_theta = config.rope_theta
        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x):
        batch, length, _ = x.shape
        positions = torch.arange(length, device=x.device)
        q = self.q_proj(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, positions, self.rope_theta)
        k = apply_rope(k, positions, self.rope_theta)
        repeat = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attended = attended.transpose(1, 2).contiguous().view(batch, length, -1)
        return self.o_proj(attended)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.ffn_dim, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.ffn_dim, bias=False)
        self.down_proj = nn.Linear(config.ffn_dim, config.d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attn = GQAAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.ffn = SwiGLU(config)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class AncientDecoder(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.config.validate()
        self.target_n_layers = self.config.n_layers
        c = self.config
        self.embed_tokens = nn.Embedding(c.vocab_size, c.d_model)
        self.layers = nn.ModuleList([TransformerBlock(c) for _ in range(c.n_layers)])
        self.final_norm = RMSNorm(c.d_model, c.rms_norm_eps)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        if c.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError(f"sequence length exceeds {self.config.max_seq_len}")
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            if self.config.activation_checkpointing and self.training and x.requires_grad:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        logits = F.linear(self.final_norm(x), self.embed_tokens.weight if self.config.tie_embeddings else self.lm_head.weight)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))
        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens: int, eos_token_id: int | None = None):
        self.eval()
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.max_seq_len:]
            logits = self(context)["logits"][:, -1, :].float()
            next_id = logits.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat((input_ids, next_id), dim=1)
            if eos_token_id is not None and torch.all(next_id == eos_token_id):
                break
        return input_ids

    def config_dict(self):
        return asdict(self.config)

    def truncate_layers(self, depth: int):
        """Keep the first ``depth`` layer objects, preserving their parameters."""
        if not 1 <= depth <= len(self.layers):
            raise ValueError(f"depth must be in [1, {len(self.layers)}]")
        self.layers = nn.ModuleList(list(self.layers[:depth]))
        self.config.n_layers = depth

    def grow_layers(self, depth: int, template_state: dict[str, torch.Tensor],
                    *, device=None, dtype=None):
        """Append initialization layers as exact identity residual blocks.

        New layers inherit their non-residual weights from the full-depth
        initialization checkpoint. Zeroing attention output and FFN down
        projections makes each inserted block an exact identity at expansion.
        Existing block objects and parameters are left untouched.
        """
        old_depth = len(self.layers)
        if depth <= old_depth:
            raise ValueError(f"new depth {depth} must exceed current depth {old_depth}")
        if depth > self.target_n_layers:
            raise ValueError(f"new depth {depth} exceeds configured target {self.target_n_layers}")
        device = device or next(self.parameters()).device
        dtype = dtype or next(self.parameters()).dtype
        added = []
        for index in range(old_depth, depth):
            prefix = f"layers.{index}."
            layer_state = {key[len(prefix):]: value
                           for key, value in template_state.items()
                           if key.startswith(prefix)}
            block = TransformerBlock(self.config)
            missing, unexpected = block.load_state_dict(layer_state, strict=False)
            if missing or unexpected:
                raise ValueError(f"invalid initialization for layer {index}: "
                                 f"missing={missing}, unexpected={unexpected}")
            block.to(device=device, dtype=dtype)
            with torch.no_grad():
                block.attn.o_proj.weight.zero_()
                block.ffn.down_proj.weight.zero_()
            self.layers.append(block)
            added.extend(block.parameters())
        self.config.n_layers = depth
        return added
