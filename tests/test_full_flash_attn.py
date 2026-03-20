# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

try:
    from vllm.vllm_flash_attn import (
        fa_version_unsupported_reason,
        flash_attn_varlen_func,
        is_fa_version_supported,
    )
except ImportError:
    if current_platform.is_rocm():
        pytest.skip(
            "vllm_flash_attn is not supported for vLLM on ROCm.",
            allow_module_level=True,
        )


NUM_HEADS = [
    (1, 1),
    (2, 1),
    (2, 2),
    (3, 1),
    (3, 3),
    (4, 1),
    (4, 4),
    (6, 1),
    (6, 2),
    (6, 3),
    (8, 1),
    (8, 2),
    (8, 4),
    (12, 1),
    (12, 3),
    (12, 4),
    (16, 1),
    (16, 2),
    (16, 4),
]
EDGE_NUM_HEADS = [(1, 1), (2, 2), (4, 1), (6, 2), (8, 4), (12, 3), (16, 2)]
HEAD_SIZES = [32, 64, 96, 192, 40, 72, 80, 128, 256]
BLOCK_SIZES = [16, 32, 64]
DTYPES = [torch.float16]
QDTYPES = [None, torch.float8_e4m3fn]
# one value large enough to test overflow in index calculation.
# one value small enough to test the schema op check
NUM_BLOCKS = [32768, 2048]
SOFT_CAPS = [None, 10.0]
SLIDING_WINDOWS = [None, 1, 32, 256, 1024]


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    causal: bool = True,
    sliding_window: int | None = None,
    window_size: tuple[int, int] | None = None,
    soft_cap: float | None = None,
    alibi_slopes: torch.Tensor | None = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: list[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len]
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        if window_size is None:
            resolved_window_size = (
                (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
            )
        else:
            resolved_window_size = window_size
        if causal:
            resolved_window_size = (resolved_window_size[0], 0)

        row_idx = torch.arange(query_len, dtype=torch.long).unsqueeze(1)
        col_idx = torch.arange(kv_len, dtype=torch.long)
        shift = kv_len - query_len
        if causal:
            mask = col_idx > (row_idx + shift)
        else:
            mask = torch.zeros(query_len, kv_len, dtype=torch.bool)
        if resolved_window_size[0] >= 0 or resolved_window_size[1] >= 0:
            left, right = resolved_window_size
            if left < 0:
                local_mask = col_idx > (row_idx + shift + right)
            else:
                upper = torch.minimum(
                    row_idx + shift + right,
                    torch.full_like(row_idx, kv_len),
                )
                local_mask = torch.logical_or(
                    col_idx > upper,
                    col_idx < (row_idx + shift - left),
                )
            mask |= local_mask
        if soft_cap is not None:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        if alibi_slopes is not None:
            slopes = alibi_slopes if alibi_slopes.ndim == 1 else alibi_slopes[i]
            row_idx = torch.arange(query_len, dtype=torch.float32).unsqueeze(1)
            col_idx = torch.arange(kv_len, dtype=torch.float32)
            relative_pos = torch.abs(row_idx + kv_len - query_len - col_idx)
            attn = attn + (-slopes.float().unsqueeze(-1).unsqueeze(-1) * relative_pos)
        mask = mask.unsqueeze(0).expand(attn.shape[0], -1, -1)
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        # Fully masked rows should return zeros instead of NaNs.
        attn = attn.masked_fill(torch.all(mask, dim=-1, keepdim=True), 0.0)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


@pytest.mark.parametrize("use_out", [True, False])
@pytest.mark.parametrize(
    "seq_lens", [[(1, 1328), (5, 18), (129, 463)], [(1, 523), (1, 37), (1, 2011)]]
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOWS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", SOFT_CAPS)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("fa_version", [2, 3])
@pytest.mark.parametrize("q_dtype", QDTYPES)
@torch.inference_mode()
def test_varlen_with_paged_kv(
    use_out: bool,
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    fa_version: int,
    q_dtype: torch.dtype | None,
) -> None:
    torch.set_default_device("cuda")
    if not is_fa_version_supported(fa_version):
        pytest.skip(
            f"Flash attention version {fa_version} not supported due "
            f'to: "{fa_version_unsupported_reason(fa_version)}"'
        )
    if q_dtype is not None and (dtype != torch.bfloat16 or fa_version == 2):
        pytest.skip(
            "Flash attention with quantized inputs is only "
            "supported on version 3 with bfloat16 base type"
        )
    set_random_seed(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    out = torch.empty_like(query) if use_out else None

    maybe_quantized_query = query
    maybe_quantized_key_cache = key_cache
    maybe_quantized_value_cache = value_cache
    q_descale = None
    k_descale = None
    v_descale = None
    if q_dtype is not None:
        # QKV are drawn from N(0, 1): no need for a fp8 scaling factor
        maybe_quantized_query = query.to(q_dtype)
        maybe_quantized_key_cache = key_cache.to(q_dtype)
        maybe_quantized_value_cache = value_cache.to(q_dtype)

        scale_shape = (num_seqs, num_kv_heads)
        q_descale = torch.ones(scale_shape, dtype=torch.float32)
        k_descale = torch.ones(scale_shape, dtype=torch.float32)
        v_descale = torch.ones(scale_shape, dtype=torch.float32)

    output = flash_attn_varlen_func(
        q=maybe_quantized_query,
        k=maybe_quantized_key_cache,
        v=maybe_quantized_value_cache,
        out=out,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
        fa_version=fa_version,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    output = output if not use_out else out

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
    )
    atol, rtol = 1.5e-2, 1e-2
    if q_dtype is not None:
        atol, rtol = 1.5e-1, 1.5e-1
    (
        torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol),
        f"{torch.max(torch.abs(output - ref_output))}",
    )


def _make_block_tables(
    num_seqs: int, max_num_blocks_per_seq: int, num_blocks: int
) -> torch.Tensor:
    base = torch.arange(max_num_blocks_per_seq, dtype=torch.int32)
    offsets = torch.arange(num_seqs, dtype=torch.int32).unsqueeze(1)
    return (base.unsqueeze(0) + offsets) % num_blocks


@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_heads", EDGE_NUM_HEADS)
@pytest.mark.parametrize("head_size", [64])
@torch.inference_mode()
def test_varlen_with_paged_kv_causal_prefix_mask(
    block_size: int,
    num_heads: tuple[int, int],
    head_size: int,
) -> None:
    torch.set_default_device("cuda")
    set_random_seed(0)

    seq_lens = [
        (block_size + 3, block_size - 1),
        (block_size, block_size),
        (1, block_size + 1),
    ]
    query_lens = [query_len for query_len, _ in seq_lens]
    kv_lens = [kv_len for _, kv_len in seq_lens]
    num_query_heads, num_kv_heads = num_heads
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    num_seqs = len(seq_lens)
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=torch.float16)
    num_blocks = 2 * ((max_kv_len + block_size - 1) // block_size) + num_seqs
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=torch.float16
    )
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    seqused_k = torch.tensor(kv_lens, dtype=torch.int32)
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = _make_block_tables(num_seqs, max_num_blocks_per_seq, num_blocks)

    output = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_query_lens,
        seqused_k=seqused_k,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        block_table=block_tables,
        fa_version=2,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
    )

    torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=1e-2)

    fully_masked_prefix = query_lens[0] - kv_lens[0]
    assert fully_masked_prefix > 0
    torch.testing.assert_close(
        output[:fully_masked_prefix],
        torch.zeros_like(output[:fully_masked_prefix]),
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("sliding_window", [1, 5])
@pytest.mark.parametrize("num_heads", EDGE_NUM_HEADS)
@pytest.mark.parametrize("head_size", [80])
@pytest.mark.parametrize("soft_cap", [None, 10.0])
@torch.inference_mode()
def test_varlen_with_paged_kv_block_boundaries(
    block_size: int,
    sliding_window: int,
    num_heads: tuple[int, int],
    head_size: int,
    soft_cap: float | None,
) -> None:
    torch.set_default_device("cuda")
    set_random_seed(0)

    seq_lens = [
        (1, block_size - 1),
        (2, block_size),
        (3, block_size + 1),
        (block_size + 1, 2 * block_size + 1),
    ]
    query_lens = [query_len for query_len, _ in seq_lens]
    kv_lens = [kv_len for _, kv_len in seq_lens]
    num_query_heads, num_kv_heads = num_heads
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    num_seqs = len(seq_lens)
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=torch.float16)
    num_blocks = 2 * ((max_kv_len + block_size - 1) // block_size) + num_seqs
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=torch.float16
    )
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    seqused_k = torch.tensor(kv_lens, dtype=torch.int32)
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = _make_block_tables(num_seqs, max_num_blocks_per_seq, num_blocks)

    output = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_query_lens,
        seqused_k=seqused_k,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=(sliding_window - 1, 0),
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
        fa_version=2,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
    )

    torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("window_size", [(0, 0), (2, 1), (5, 3)])
@pytest.mark.parametrize("num_heads", [(4, 1), (6, 2), (8, 4)])
@pytest.mark.parametrize("head_size", [64])
@pytest.mark.parametrize("soft_cap", [None, 10.0])
@torch.inference_mode()
def test_varlen_with_paged_kv_local_window_lr(
    block_size: int,
    window_size: tuple[int, int],
    num_heads: tuple[int, int],
    head_size: int,
    soft_cap: float | None,
) -> None:
    torch.set_default_device("cuda")
    set_random_seed(0)

    seq_lens = [
        (3, block_size - 1),
        (block_size + 1, block_size + 2),
        (block_size - 2, 2 * block_size + 1),
    ]
    query_lens = [query_len for query_len, _ in seq_lens]
    kv_lens = [kv_len for _, kv_len in seq_lens]
    num_query_heads, num_kv_heads = num_heads
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    num_seqs = len(seq_lens)
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=torch.float16)
    num_blocks = 2 * ((max_kv_len + block_size - 1) // block_size) + num_seqs
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=torch.float16
    )
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    seqused_k = torch.tensor(kv_lens, dtype=torch.int32)
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = _make_block_tables(num_seqs, max_num_blocks_per_seq, num_blocks)

    output = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_query_lens,
        seqused_k=seqused_k,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=False,
        window_size=window_size,
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
        fa_version=2,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        causal=False,
        window_size=window_size,
        soft_cap=soft_cap,
    )

    torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("num_heads", [(4, 1), (8, 2), (12, 3)])
@pytest.mark.parametrize("head_size", [64, 80])
@pytest.mark.parametrize("soft_cap", [None, 10.0])
@torch.inference_mode()
def test_varlen_with_paged_kv_alibi(
    block_size: int,
    causal: bool,
    num_heads: tuple[int, int],
    head_size: int,
    soft_cap: float | None,
) -> None:
    torch.set_default_device("cuda")
    set_random_seed(0)

    seq_lens = [
        (block_size - 1, block_size + 1),
        (block_size + 2, 2 * block_size + 1),
        (2, block_size),
    ]
    query_lens = [query_len for query_len, _ in seq_lens]
    kv_lens = [kv_len for _, kv_len in seq_lens]
    num_query_heads, num_kv_heads = num_heads
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    num_seqs = len(seq_lens)
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=torch.float16)
    num_blocks = 2 * ((max_kv_len + block_size - 1) // block_size) + num_seqs
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=torch.float16
    )
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    seqused_k = torch.tensor(kv_lens, dtype=torch.int32)
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = _make_block_tables(num_seqs, max_num_blocks_per_seq, num_blocks)
    alibi_slopes = torch.rand(num_seqs, num_query_heads, dtype=torch.float32) * 0.3

    output = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_query_lens,
        seqused_k=seqused_k,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=causal,
        block_table=block_tables,
        alibi_slopes=alibi_slopes,
        softcap=soft_cap if soft_cap is not None else 0,
        fa_version=2,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        causal=causal,
        soft_cap=soft_cap,
        alibi_slopes=alibi_slopes,
    )

    torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=1e-2)
