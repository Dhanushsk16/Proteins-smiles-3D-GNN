"""SMILES Transformer encoder (Phase 2 sequence branch).

A pre-normalization Transformer encoder over SMILES tokens, built to mirror the
PeptideCLM-2 architecture so pretrained checkpoints can be loaded into it. It is
exposed behind the same config-string interface as the 3D GNN encoders in
``model.py`` (select with ``ENCODER_TYPE = "smiles"`` / ``"smiles_base"`` / ...).

This step implements the encoder shell only: tokenizer loading, the architecture,
size presets, the forward contract, and pretrained-weight loading. Training,
masking, descriptor heads, and cross-modal code are intentionally out of scope.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Hugging Face hub id for the pretrained tokenizer. We load its vocabulary rather
# than building a new one.
#
# NOTE: The task spec named "aaronfeller/PeptideMTR", which is not a live repo id.
# The published PeptideCLM-2 MTR family lives under "aaronfeller/peptideclm-2-mtr-*"
# and shares one 405-token tokenizer (pad 0, unk 1, cls 2, sep 3, mask 4), so we
# load the tokenizer from the -small repo for every size preset.
TOKENIZER_HUB_ID = "aaronfeller/peptideclm-2-mtr-small"

VOCAB_SIZE = 405
MAX_SEQ_LEN = 2048

# Special token ids, fixed by the pretrained tokenizer.
PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
CLS_TOKEN_ID = 2
SEP_TOKEN_ID = 3
MASK_TOKEN_ID = 4

# Size presets, selectable by string.
SMILES_PRESETS = {
    "small": dict(embed_dim=512, num_heads=8, num_blocks=14, ffn_hidden_dim=768),
    "base": dict(embed_dim=768, num_heads=12, num_blocks=24, ffn_hidden_dim=1024),
    "large": dict(embed_dim=1024, num_heads=16, num_blocks=32, ffn_hidden_dim=2048),
}


def load_tokenizer():
    """Load the pretrained PeptideMTR tokenizer with ``model_max_length`` set to 2048."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_HUB_ID)
    tokenizer.model_max_length = MAX_SEQ_LEN
    return tokenizer


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with rotary positional embeddings.

    A single fused, bias-free linear projection produces query, key, and value.
    RoPE is applied to the query and key projections (rotary dim == head dim).
    A padding mask (1 = real, 0 = pad) becomes an additive bias that drives padded
    key positions to a large negative value before the softmax.
    """

    def __init__(self, embed_dim, num_heads, max_seq_len=MAX_SEQ_LEN, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        from torchtune.modules import RotaryPositionalEmbeddings
        # Rotary dimension equals the per-head dimension.
        self.rope = RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x, padding_mask=None):
        bsz, seq_len, _ = x.shape

        qkv = self.qkv_proj(x)  # (B, S, 3E)
        qkv = qkv.view(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, S, H, Dh)

        # torchtune RoPE expects (B, S, H, Dh) and returns the same shape.
        q = self.rope(q)
        k = self.rope(k)

        # Move heads forward for scaled dot-product attention: (B, H, S, Dh).
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_bias = None
        if padding_mask is not None:
            # padded key positions (mask == 0) get a large negative additive bias.
            pad = (padding_mask == 0)  # (B, S), True where padding
            attn_bias = torch.zeros(bsz, 1, 1, seq_len, dtype=q.dtype, device=q.device)
            attn_bias = attn_bias.masked_fill(
                pad[:, None, None, :], torch.finfo(q.dtype).min
            )

        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias, dropout_p=dropout_p
        )  # (B, H, S, Dh)

        out = out.transpose(1, 2).reshape(bsz, seq_len, self.embed_dim)
        return self.out_proj(out)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network.

    A single linear maps ``embed_dim`` to ``2 * ffn_hidden_dim``; the halves are
    combined as ``SiLU(a) * b`` and projected back to ``embed_dim``.
    """

    def __init__(self, embed_dim, ffn_hidden_dim, dropout=0.1):
        super().__init__()
        self.w_in = nn.Linear(embed_dim, 2 * ffn_hidden_dim)
        self.w_out = nn.Linear(ffn_hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        a, b = self.w_in(x).chunk(2, dim=-1)
        x = F.silu(a) * b
        return self.dropout(self.w_out(x))


class TransformerBlock(nn.Module):
    """Pre-norm block: x = x + Attn(LN(x)); x = x + SwiGLU(LN(x))."""

    def __init__(self, embed_dim, num_heads, ffn_hidden_dim, max_seq_len=MAX_SEQ_LEN, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, max_seq_len, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = SwiGLU(embed_dim, ffn_hidden_dim, dropout)

    def forward(self, x, padding_mask=None):
        x = x + self.attn(self.norm1(x), padding_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class SMILESTransformerEncoder(nn.Module):
    """Pre-normalization Transformer encoder over SMILES tokens.

    Forward returns three first-class values:
      * per-token hidden states, ``(B, S, embed_dim)``
      * masked-language-model logits, ``(B, S, vocab_size)``
      * mean-pooled sequence embedding over real tokens, ``(B, embed_dim)``
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_blocks,
        ffn_hidden_dim,
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        dropout=0.1,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.ffn_hidden_dim = ffn_hidden_dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Token embedding only: no positional or token-type embedding layers.
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN_ID)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_hidden_dim, max_seq_len, dropout)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)

        # Sequence head for later masked-token prediction.
        self.sequence_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids, padding_mask):
        """
        Args:
            input_ids: LongTensor ``(B, S)`` of token ids.
            padding_mask: Tensor ``(B, S)`` where 1 marks real tokens, 0 marks padding.

        Returns:
            hidden_states: ``(B, S, embed_dim)``
            logits: ``(B, S, vocab_size)``
            pooled: ``(B, embed_dim)`` mean over real token positions only.
        """
        hidden = self.token_embedding(input_ids)
        for block in self.blocks:
            hidden = block(hidden, padding_mask)
        hidden = self.final_norm(hidden)

        logits = self.sequence_head(hidden)

        # Mean-pool over real (non-padded) positions only.
        mask = padding_mask.unsqueeze(-1).to(hidden.dtype)  # (B, S, 1)
        summed = (hidden * mask).sum(dim=1)                 # (B, E)
        counts = mask.sum(dim=1).clamp(min=1.0)             # (B, 1)
        pooled = summed / counts

        return hidden, logits, pooled

    def load_pretrained_weights(self, state_dict, strict=True):
        """Load pretrained encoder weights (e.g. a PeptideCLM-2 checkpoint).

        Reports any missing or unexpected keys instead of failing silently. With
        ``strict=True`` a mismatch raises after reporting.

        Returns:
            (missing_keys, unexpected_keys)
        """
        incompatible = self.load_state_dict(state_dict, strict=False)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)

        if missing:
            print(f"[load_pretrained_weights] {len(missing)} missing key(s): {missing}")
        if unexpected:
            print(f"[load_pretrained_weights] {len(unexpected)} unexpected key(s): {unexpected}")
        if not missing and not unexpected:
            print("[load_pretrained_weights] all keys matched exactly.")

        if strict and (missing or unexpected):
            raise RuntimeError(
                f"Strict load failed: {len(missing)} missing, "
                f"{len(unexpected)} unexpected keys."
            )
        return missing, unexpected


def get_smiles_encoder(size="small", **overrides):
    """Factory: build a SMILES encoder from a size-preset string.

    Args:
        size: one of ``"small"``, ``"base"``, ``"large"``.
        **overrides: optional keyword overrides of preset values.
    """
    key = size.lower()
    if key not in SMILES_PRESETS:
        raise ValueError(
            f"Unknown SMILES size preset: {size!r}. "
            f"Choose from {list(SMILES_PRESETS)}."
        )
    cfg = dict(SMILES_PRESETS[key])
    cfg.update(overrides)
    return SMILESTransformerEncoder(
        vocab_size=VOCAB_SIZE, max_seq_len=MAX_SEQ_LEN, **cfg
    )


# --------------------------------------------------------------------------- #
# Acceptance self-test
# --------------------------------------------------------------------------- #

def _self_test():
    import numpy as np

    torch.manual_seed(0)
    model = get_smiles_encoder("small")
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Small config parameter count: {num_params:,}")
    assert 31_000_000 <= num_params <= 32_000_000, (
        f"Expected ~31-32M params for Small, got {num_params:,}"
    )

    # Two example peptide SMILES of different lengths.
    peptides = [
        "CC(C)C[C@@H](C(=O)O)N",                       # leucine (short)
        "N[C@@H](CC1=CC=CC=C1)C(=O)N[C@@H](CO)C(=O)O", # Phe-Ser dipeptide (longer)
    ]

    tokenizer = load_tokenizer()
    encoded = tokenizer(
        peptides,
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    padding_mask = encoded["attention_mask"]  # 1 real, 0 pad
    seq_len = input_ids.shape[1]
    print(f"Batch shape: input_ids={tuple(input_ids.shape)}, "
          f"padding_mask={tuple(padding_mask.shape)}")

    with torch.no_grad():
        hidden, logits, pooled = model(input_ids, padding_mask)

    assert hidden.shape == (2, seq_len, 512), hidden.shape
    assert logits.shape == (2, seq_len, VOCAB_SIZE), logits.shape
    assert pooled.shape == (2, 512), pooled.shape
    print(f"hidden={tuple(hidden.shape)}, logits={tuple(logits.shape)}, "
          f"pooled={tuple(pooled.shape)}  OK")

    # Padding invariance: changing only the padded region must not move the pooled
    # embedding. The shorter sequence has at least one padded position.
    pad_positions = (padding_mask == 0)
    assert pad_positions.any(), "Test needs at least one padded position; add a longer peptide."

    corrupted = input_ids.clone()
    # Overwrite padded positions with an arbitrary different token id.
    corrupted[pad_positions] = MASK_TOKEN_ID
    with torch.no_grad():
        _, _, pooled_corrupted = model(corrupted, padding_mask)

    max_diff = (pooled - pooled_corrupted).abs().max().item()
    print(f"Max pooled diff after corrupting padding region: {max_diff:.3e}")
    assert torch.allclose(pooled, pooled_corrupted, atol=1e-5), (
        f"Pooled embedding changed when only padding was altered (max diff {max_diff:.3e})"
    )

    print("Self-test PASSED.")


if __name__ == "__main__":
    _self_test()
