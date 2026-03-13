import math
import os
import sys
from dataclasses import dataclass
from typing import List

import torch

# Avoid local source package shadowing installed wheel.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path = [
    p for p in sys.path
    if os.path.abspath(p if p else os.getcwd()) != _REPO_ROOT
]

try:
    from vllm_flash_attn import sparse_attn_func
except ImportError as e:
    print("[ERR] failed to import vllm_flash_attn.sparse_attn_func:", e)
    sys.exit(1)


@dataclass
class SparseCase:
    case_id: int
    name: str
    batch: int
    seqlen_q: int
    seqlen_k: int
    nheads: int
    headdim: int
    causal: bool
    nnz_s: int
    nnz_v: int
    pattern: str  # block_only | column_only | mixed


def _make_patterns(cfg: SparseCase, device: torch.device):
    assert cfg.headdim == 128, "sparse kernel currently only supports headdim=128"
    block_m = 64
    block_n = 64
    num_rows = (cfg.seqlen_q + block_m - 1) // block_m

    block_count = torch.zeros((cfg.batch, cfg.nheads, num_rows), dtype=torch.int32, device=device)
    block_offset = torch.zeros((cfg.batch, cfg.nheads, num_rows, cfg.nnz_s), dtype=torch.int32, device=device)
    column_count = torch.zeros((cfg.batch, cfg.nheads, num_rows), dtype=torch.int32, device=device)
    column_index = torch.zeros((cfg.batch, cfg.nheads, num_rows, cfg.nnz_v), dtype=torch.int32, device=device)

    g = torch.Generator(device=device)
    g.manual_seed(20260310 + cfg.case_id)

    dense_block_starts = list(range(0, cfg.seqlen_k, block_n))

    for b in range(cfg.batch):
        for h in range(cfg.nheads):
            for r in range(num_rows):
                if cfg.pattern == "block_only":
                    starts = dense_block_starts[: cfg.nnz_s]
                    cols: List[int] = []
                elif cfg.pattern == "column_only":
                    starts = []
                    perm = torch.randperm(cfg.seqlen_k, generator=g, device=device)
                    cols = perm[: cfg.nnz_v].tolist()
                    cols.sort()
                elif cfg.pattern == "mixed":
                    starts = dense_block_starts[: max(1, min(len(dense_block_starts), cfg.nnz_s - 1))]
                    # Add some extra random columns not guaranteed to align with blocks.
                    perm = torch.randperm(cfg.seqlen_k, generator=g, device=device)
                    cols = perm[: max(1, cfg.nnz_v // 2)].tolist()
                    cols.sort()
                else:
                    raise ValueError(f"unknown pattern: {cfg.pattern}")

                bc = min(len(starts), cfg.nnz_s)
                vc = min(len(cols), cfg.nnz_v)

                block_count[b, h, r] = bc
                column_count[b, h, r] = vc

                if bc > 0:
                    block_offset[b, h, r, :bc] = torch.tensor(starts[:bc], dtype=torch.int32, device=device)
                if vc > 0:
                    column_index[b, h, r, :vc] = torch.tensor(cols[:vc], dtype=torch.int32, device=device)

    return block_count, block_offset, column_count, column_index


def _build_sparse_mask(
    block_count,
    block_offset,
    column_count,
    column_index,
    seqlen_q: int,
    seqlen_k: int,
    block_m: int = 64,
    block_n: int = 64,
):
    # [B, H, Sq, Sk]
    bsz, nheads, num_rows = block_count.shape
    mask = torch.zeros((bsz, nheads, seqlen_q, seqlen_k), dtype=torch.bool, device=block_count.device)

    for b in range(bsz):
        for h in range(nheads):
            for r in range(num_rows):
                rs = r * block_m
                re = min((r + 1) * block_m, seqlen_q)
                if rs >= re:
                    continue

                allowed = torch.zeros((seqlen_k,), dtype=torch.bool, device=mask.device)

                bc = int(block_count[b, h, r].item())
                if bc > 0:
                    starts = block_offset[b, h, r, :bc].tolist()
                    for s in starts:
                        s0 = max(0, int(s))
                        s1 = min(seqlen_k, s0 + block_n)
                        if s0 < s1:
                            allowed[s0:s1] = True

                vc = int(column_count[b, h, r].item())
                if vc > 0:
                    cols = column_index[b, h, r, :vc]
                    cols = cols[(cols >= 0) & (cols < seqlen_k)]
                    if cols.numel() > 0:
                        allowed[cols.long()] = True

                mask[b, h, rs:re, :] = allowed.view(1, -1)

    return mask


def _reference_sparse_attention(
    q,
    k,
    v,
    block_count,
    block_offset,
    column_count,
    column_index,
    causal: bool,
):
    # q/k/v: [B, Sq/Sk, H, D], Hk==H for this test
    bsz, seqlen_q, nheads, d = q.shape
    seqlen_k = k.shape[1]
    scale = 1.0 / math.sqrt(d)

    qf = q.float().permute(0, 2, 1, 3).contiguous()  # [B, H, Sq, D]
    kf = k.float().permute(0, 2, 1, 3).contiguous()  # [B, H, Sk, D]
    vf = v.float().permute(0, 2, 1, 3).contiguous()  # [B, H, Sk, D]

    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale  # [B, H, Sq, Sk]

    sparse_mask = _build_sparse_mask(
        block_count,
        block_offset,
        column_count,
        column_index,
        seqlen_q,
        seqlen_k,
    )

    if causal:
        q_idx = torch.arange(seqlen_q, device=q.device).view(1, 1, seqlen_q, 1)
        k_idx = torch.arange(seqlen_k, device=q.device).view(1, 1, 1, seqlen_k)
        shift = seqlen_k - seqlen_q
        causal_mask = k_idx <= (q_idx + shift)
        sparse_mask = sparse_mask & causal_mask

    neg_inf = torch.tensor(float("-inf"), device=q.device, dtype=scores.dtype)
    scores = torch.where(sparse_mask, scores, neg_inf)

    row_max = scores.max(dim=-1, keepdim=True).values
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    exp_scores = torch.exp(scores - row_max) * sparse_mask
    denom = exp_scores.sum(dim=-1, keepdim=True)
    probs = torch.where(denom > 0, exp_scores / denom, torch.zeros_like(exp_scores))

    out = torch.matmul(probs, vf)  # [B, H, Sq, D]
    return out.permute(0, 2, 1, 3).contiguous().to(dtype=q.dtype)


def _run_case(cfg: SparseCase, device: torch.device):
    torch.manual_seed(310000 + cfg.case_id)
    torch.cuda.manual_seed_all(310000 + cfg.case_id)

    q = torch.randn(cfg.batch, cfg.seqlen_q, cfg.nheads, cfg.headdim, dtype=torch.float16, device=device)
    k = torch.randn(cfg.batch, cfg.seqlen_k, cfg.nheads, cfg.headdim, dtype=torch.float16, device=device)
    v = torch.randn(cfg.batch, cfg.seqlen_k, cfg.nheads, cfg.headdim, dtype=torch.float16, device=device)

    block_count, block_offset, column_count, column_index = _make_patterns(cfg, device)

    out_sparse = sparse_attn_func(
        q,
        k,
        v,
        block_count,
        block_offset,
        column_count,
        column_index,
        dropout_p=0.0,
        softmax_scale=None,
        causal=cfg.causal,
        return_attn_probs=False,
    )

    out_ref = _reference_sparse_attention(
        q,
        k,
        v,
        block_count,
        block_offset,
        column_count,
        column_index,
        causal=cfg.causal,
    )

    diff = (out_sparse - out_ref).abs().float()
    max_diff = float(diff.max().item())
    mean_diff = float(diff.mean().item())

    # fp16 + sparse gather path tolerance
    ok = max_diff <= 8e-2 and mean_diff <= 8e-3
    status = "PASS" if ok else "FAIL"
    print(
        f"[{status}] case={cfg.case_id:02d} name={cfg.name} "
        f"max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}"
    )
    return ok


def main():
    if not torch.cuda.is_available():
        print("[ERR] CUDA is required")
        sys.exit(1)

    device = torch.device("cuda")

    cases = [
        SparseCase(
            case_id=1,
            name="block_only_dense_cover_noncausal",
            batch=2,
            seqlen_q=128,
            seqlen_k=192,
            nheads=4,
            headdim=128,
            causal=False,
            nnz_s=4,
            nnz_v=64,
            pattern="block_only",
        ),
        SparseCase(
            case_id=2,
            name="column_only_noncausal",
            batch=1,
            seqlen_q=96,
            seqlen_k=160,
            nheads=4,
            headdim=128,
            causal=False,
            nnz_s=4,
            nnz_v=48,
            pattern="column_only",
        ),
        SparseCase(
            case_id=3,
            name="mixed_causal",
            batch=1,
            seqlen_q=128,
            seqlen_k=128,
            nheads=8,
            headdim=128,
            causal=True,
            nnz_s=3,
            nnz_v=48,
            pattern="mixed",
        ),
    ]

    only = os.environ.get("CASE_IDS", "").strip()
    if only:
        wanted = {int(x) for x in only.split(",") if x.strip()}
        cases = [c for c in cases if c.case_id in wanted]

    total = len(cases)
    if total == 0:
        print("[ERR] no cases selected")
        sys.exit(1)

    passed = 0
    for cfg in cases:
        ok = _run_case(cfg, device)
        if ok:
            passed += 1

    print(f"\nSparse test summary: {passed}/{total} PASS")
    if passed != total:
        sys.exit(1)


if __name__ == "__main__":
    main()
