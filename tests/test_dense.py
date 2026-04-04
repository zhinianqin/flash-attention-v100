import math
import os
import sys
import traceback
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Set, Tuple

import torch

try:
    import pytest
except ImportError:
    pytest = None

# 避免仓库内源码包(vllm_flash_attn/)覆盖已安装wheel，导致找不到编译后的 .so
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path = [
    p for p in sys.path
    if os.path.abspath(p if p else os.getcwd()) != _REPO_ROOT
]

HEAD_DIMS = (32, 64, 96, 128, 192, 256)
_VLLM_FLASH_ATTN_LOADED = False
_VLLM_FLASH_ATTN_IMPORT_ERROR: Optional[ImportError] = None


@dataclass
class DenseCaseConfig:
    case_id: int
    suite: str  # numerical | splitkv
    seqlen_q: int
    seqlen_k: int
    num_heads: int
    num_heads_k: int
    split_num_splits: int
    causal: bool
    use_dropout: bool
    use_local: bool
    use_alibi: bool
    use_varlen: bool


@dataclass
class DenseCaseResult:
    cfg: DenseCaseConfig
    head_dim: int
    status: str
    detail: str
    max_diff: Optional[float] = None
    mean_diff: Optional[float] = None


@dataclass
class SmokeCaseResult:
    head_dim: int
    status: str
    detail: str
    out_max_diff: Optional[float] = None
    out_mean_diff: Optional[float] = None
    lse_max_diff: Optional[float] = None
    lse_mean_diff: Optional[float] = None


def try_import_vllm_flash_attn() -> Tuple[bool, Optional[str]]:
    global _VLLM_FLASH_ATTN_LOADED
    global _VLLM_FLASH_ATTN_IMPORT_ERROR

    if _VLLM_FLASH_ATTN_LOADED:
        return True, None
    if _VLLM_FLASH_ATTN_IMPORT_ERROR is not None:
        return False, str(_VLLM_FLASH_ATTN_IMPORT_ERROR)

    try:
        import vllm_flash_attn  # noqa: F401
    except ImportError as exc:
        _VLLM_FLASH_ATTN_IMPORT_ERROR = exc
        return False, str(exc)

    _VLLM_FLASH_ATTN_LOADED = True
    return True, None


def get_runtime_error_message() -> Optional[str]:
    if not torch.cuda.is_available():
        return "需要 GPU 环境"

    ok, detail = try_import_vllm_flash_attn()
    if not ok:
        base_msg = "无法导入 vllm_flash_attn，请检查是否已执行 pip install vllm-flash-attn"
        return base_msg if not detail else f"{base_msg}: {detail}"

    if not hasattr(torch.ops, "_vllm_fa2_C") or not hasattr(torch.ops._vllm_fa2_C, "varlen_fwd"):
        return "已导入 vllm_flash_attn，但找不到 torch.ops._vllm_fa2_C.varlen_fwd"

    return None


def ensure_runtime_available(for_pytest: bool = False) -> None:
    error_message = get_runtime_error_message()
    if error_message is None:
        return
    if for_pytest:
        if pytest is None:
            raise RuntimeError(error_message)
        pytest.skip(error_message)
    raise RuntimeError(error_message)


def parse_csv_ints(env_name: str, raw_value: str) -> List[int]:
    items = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{env_name} 不能为空")

    values: List[int] = []
    for item in items:
        try:
            values.append(int(item))
        except ValueError as exc:
            raise ValueError(f"{env_name} 包含非法整数: {item}") from exc
    return values


def get_selected_head_dims() -> List[int]:
    raw_value = os.environ.get("HEAD_DIMS", "").strip()
    if not raw_value:
        return list(HEAD_DIMS)

    seen: Set[int] = set()
    selected: List[int] = []
    allowed = set(HEAD_DIMS)
    invalid: List[int] = []

    for head_dim in parse_csv_ints("HEAD_DIMS", raw_value):
        if head_dim not in allowed:
            invalid.append(head_dim)
            continue
        if head_dim in seen:
            continue
        seen.add(head_dim)
        selected.append(head_dim)

    if invalid:
        raise ValueError(
            f"HEAD_DIMS 仅支持 {list(HEAD_DIMS)} 的子集，收到非法值: {invalid}"
        )
    if not selected:
        raise ValueError("HEAD_DIMS 过滤后为空，请至少保留一个合法 head_dim")
    return selected


def get_dense_suite() -> str:
    dense_suite = os.environ.get("DENSE_SUITE", "all").strip().lower()
    if dense_suite not in {"all", "numerical", "splitkv"}:
        raise ValueError("DENSE_SUITE 仅支持: all | numerical | splitkv")
    return dense_suite


def get_case_id_filter() -> Optional[Set[int]]:
    raw_value = os.environ.get("CASE_IDS", "").strip()
    if not raw_value:
        return None
    return set(parse_csv_ints("CASE_IDS", raw_value))


def build_cu_seqlens(lengths: List[int], device: str) -> torch.Tensor:
    cu = [0]
    for x in lengths:
        cu.append(cu[-1] + int(x))
    return torch.tensor(cu, dtype=torch.int32, device=device)


def make_mask(
    seqlen_q: int,
    seqlen_k: int,
    causal: bool,
    window_size: Tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    q_idx = torch.arange(seqlen_q, device=device).unsqueeze(1)
    k_idx = torch.arange(seqlen_k, device=device).unsqueeze(0)
    shift = seqlen_k - seqlen_q

    keep = torch.ones((seqlen_q, seqlen_k), dtype=torch.bool, device=device)

    win_left, win_right = window_size
    if causal:
        keep &= k_idx <= (q_idx + shift)
        win_right = 0

    if win_left >= 0:
        keep &= k_idx >= (q_idx + shift - win_left)
    if win_right >= 0:
        keep &= k_idx <= (q_idx + shift + win_right)

    return keep


def masked_softmax(scores: torch.Tensor, mask_qk: torch.Tensor) -> torch.Tensor:
    # scores: [H, Sq, Sk], mask_qk: [Sq, Sk]
    mask = mask_qk.unsqueeze(0)
    neg_inf = torch.tensor(float("-inf"), device=scores.device, dtype=scores.dtype)
    scores = torch.where(mask, scores, neg_inf)

    row_max = torch.max(scores, dim=-1, keepdim=True).values
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))

    exp_scores = torch.exp(scores - row_max) * mask
    denom = exp_scores.sum(dim=-1, keepdim=True)
    probs = torch.where(denom > 0, exp_scores / denom, torch.zeros_like(exp_scores))
    return probs


def reference_varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    num_heads: int,
    num_heads_k: int,
    head_dim: int,
    causal: bool,
    window_size: Tuple[int, int],
    alibi_slopes: Optional[torch.Tensor],
) -> torch.Tensor:
    device = q.device
    softmax_scale = 1.0 / math.sqrt(head_dim)

    batch_size = cu_seqlens_q.numel() - 1
    out = torch.empty_like(q)

    for b in range(batch_size):
        q_start = int(cu_seqlens_q[b].item())
        q_end = int(cu_seqlens_q[b + 1].item())
        k_start = int(cu_seqlens_k[b].item())
        k_end = int(cu_seqlens_k[b + 1].item())

        q_b = q[q_start:q_end].float()  # [Sq, H, D]
        k_b = k[k_start:k_end].float()  # [Sk, Hk, D]
        v_b = v[k_start:k_end].float()  # [Sk, Hk, D]

        sq = q_b.shape[0]
        sk = k_b.shape[0]

        if sq == 0:
            continue
        if sk == 0:
            out[q_start:q_end] = torch.zeros((sq, num_heads, head_dim), device=device, dtype=q.dtype)
            continue

        if num_heads_k != num_heads:
            if num_heads % num_heads_k != 0:
                raise ValueError(f"num_heads ({num_heads}) must be divisible by num_heads_k ({num_heads_k})")
            repeat = num_heads // num_heads_k
            k_b = k_b.repeat_interleave(repeat, dim=1)
            v_b = v_b.repeat_interleave(repeat, dim=1)

        scores = torch.einsum("qhd,khd->hqk", q_b, k_b) * softmax_scale

        if alibi_slopes is not None:
            slopes_b = alibi_slopes[b] if alibi_slopes.dim() == 2 else alibi_slopes
            q_idx = torch.arange(sq, device=device, dtype=torch.float32).view(1, sq, 1)
            k_idx = torch.arange(sk, device=device, dtype=torch.float32).view(1, 1, sk)
            dist = torch.abs(q_idx + (sk - sq) - k_idx)
            scores = scores + (-slopes_b.float().view(num_heads, 1, 1) * dist)

        mask = make_mask(sq, sk, causal=causal, window_size=window_size, device=device)
        probs = masked_softmax(scores, mask)
        out_b = torch.einsum("hqk,khd->qhd", probs, v_b).to(dtype=q.dtype)
        out[q_start:q_end] = out_b

    return out


def reference_varlen_attention_and_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    num_heads: int,
    num_heads_k: int,
    head_dim: int,
    causal: bool,
    window_size: Tuple[int, int],
    alibi_slopes: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = q.device
    softmax_scale = 1.0 / math.sqrt(head_dim)

    batch_size = cu_seqlens_q.numel() - 1
    out = torch.empty_like(q)
    lse = torch.empty((num_heads, q.shape[0]), dtype=torch.float32, device=device)

    for b in range(batch_size):
        q_start = int(cu_seqlens_q[b].item())
        q_end = int(cu_seqlens_q[b + 1].item())
        k_start = int(cu_seqlens_k[b].item())
        k_end = int(cu_seqlens_k[b + 1].item())

        q_b = q[q_start:q_end].float()
        k_b = k[k_start:k_end].float()
        v_b = v[k_start:k_end].float()

        sq = q_b.shape[0]
        sk = k_b.shape[0]

        if sq == 0:
            continue
        if sk == 0:
            out[q_start:q_end] = torch.zeros((sq, num_heads, head_dim), device=device, dtype=q.dtype)
            lse[:, q_start:q_end] = float("inf")
            continue

        if num_heads_k != num_heads:
            if num_heads % num_heads_k != 0:
                raise ValueError(f"num_heads ({num_heads}) must be divisible by num_heads_k ({num_heads_k})")
            repeat = num_heads // num_heads_k
            k_b = k_b.repeat_interleave(repeat, dim=1)
            v_b = v_b.repeat_interleave(repeat, dim=1)

        scores = torch.einsum("qhd,khd->hqk", q_b, k_b) * softmax_scale

        if alibi_slopes is not None:
            slopes_b = alibi_slopes[b] if alibi_slopes.dim() == 2 else alibi_slopes
            q_idx = torch.arange(sq, device=device, dtype=torch.float32).view(1, sq, 1)
            k_idx = torch.arange(sk, device=device, dtype=torch.float32).view(1, 1, sk)
            dist = torch.abs(q_idx + (sk - sq) - k_idx)
            scores = scores + (-slopes_b.float().view(num_heads, 1, 1) * dist)

        mask = make_mask(sq, sk, causal=causal, window_size=window_size, device=device).unsqueeze(0)
        neg_inf = torch.tensor(float("-inf"), device=device, dtype=scores.dtype)
        masked_scores = torch.where(mask, scores, neg_inf)
        row_max = torch.max(masked_scores, dim=-1, keepdim=True).values
        row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
        exp_scores = torch.exp(masked_scores - row_max) * mask
        denom = exp_scores.sum(dim=-1, keepdim=True)
        probs = torch.where(denom > 0, exp_scores / denom, torch.zeros_like(exp_scores))

        out_b = torch.einsum("hqk,khd->qhd", probs, v_b).to(dtype=q.dtype)
        out[q_start:q_end] = out_b

        lse_b = row_max.squeeze(-1) + torch.log(denom.squeeze(-1))
        lse_b = torch.where(denom.squeeze(-1) > 0, lse_b, torch.full_like(lse_b, float("inf")))
        lse[:, q_start:q_end] = lse_b

    return out, lse


def gen_lengths(base: int, use_varlen: bool) -> List[int]:
    if not use_varlen:
        return [base]
    return [base, max(1, base // 2), max(1, base - 7)]


def prepare_case_tensors(
    cfg: DenseCaseConfig,
    device: str,
    head_dim: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
    Tuple[int, int],
    float,
    Optional[torch.Tensor],
]:
    q_lens = gen_lengths(cfg.seqlen_q, cfg.use_varlen)
    k_lens = gen_lengths(cfg.seqlen_k, cfg.use_varlen)

    total_q = sum(q_lens)
    total_k = sum(k_lens)

    q = torch.randn(total_q, cfg.num_heads, head_dim, dtype=torch.float16, device=device).contiguous()
    k = torch.randn(total_k, cfg.num_heads_k, head_dim, dtype=torch.float16, device=device).contiguous()
    v = torch.randn(total_k, cfg.num_heads_k, head_dim, dtype=torch.float16, device=device).contiguous()

    cu_seqlens_q = build_cu_seqlens(q_lens, device)
    cu_seqlens_k = build_cu_seqlens(k_lens, device)

    max_seqlen_q = max(q_lens)
    max_seqlen_k = max(k_lens)

    if cfg.use_local:
        win_left = max(1, max_seqlen_k // 4)
        win_right = max(1, max_seqlen_k // 5)
        window_size = (win_left, win_right)
    else:
        window_size = (-1, -1)

    dropout_p = 0.17 if cfg.use_dropout else 0.0

    if cfg.use_alibi:
        batch_size = len(q_lens)
        alibi_slopes = torch.rand(batch_size, cfg.num_heads, dtype=torch.float32, device=device) * 0.3
    else:
        alibi_slopes = None

    return (
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        window_size,
        dropout_p,
        alibi_slopes,
    )


def run_numerical_case(cfg: DenseCaseConfig, device: str, head_dim: int) -> DenseCaseResult:
    (
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        window_size,
        dropout_p,
        alibi_slopes,
    ) = prepare_case_tensors(cfg, device, head_dim)

    out = torch.zeros_like(q)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    ret = torch.ops._vllm_fa2_C.varlen_fwd(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        None,
        None,
        None,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        False,
        cfg.causal,
        window_size[0],
        window_size[1],
        0.0,
        False,
        0,
        None,
    )
    if isinstance(ret, (list, tuple)) and len(ret) > 0:
        out = ret[0]

    torch.cuda.synchronize()

    if not torch.isfinite(out).all():
        return DenseCaseResult(
            cfg=cfg,
            head_dim=head_dim,
            status="FAIL",
            detail="kernel_output_has_nan_or_inf",
        )

    if dropout_p > 0.0:
        return DenseCaseResult(
            cfg=cfg,
            head_dim=head_dim,
            status="PASS",
            detail="dropout_run_ok_finite",
        )

    with torch.no_grad():
        ref_out = reference_varlen_attention(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            num_heads=cfg.num_heads,
            num_heads_k=cfg.num_heads_k,
            head_dim=head_dim,
            causal=cfg.causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
        )

    diff = torch.abs(out - ref_out)
    max_diff = float(diff.max().item())
    mean_diff = float(diff.mean().item())

    pass_cond = (max_diff <= 8e-2) and (mean_diff <= 8e-3)

    return DenseCaseResult(
        cfg=cfg,
        head_dim=head_dim,
        status="PASS" if pass_cond else "FAIL",
        detail="numerical_check",
        max_diff=max_diff,
        mean_diff=mean_diff,
    )


def run_splitkv_case(cfg: DenseCaseConfig, device: str, head_dim: int) -> DenseCaseResult:
    base_num_splits = int(os.environ.get("BASE_NUM_SPLITS", "1"))
    split_num_splits = int(os.environ.get("SPLIT_NUM_SPLITS", str(cfg.split_num_splits)))

    (
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        window_size,
        dropout_p,
        alibi_slopes,
    ) = prepare_case_tensors(cfg, device, head_dim)

    should_hit_splitkv = (
        max_seqlen_q == 1
        and cfg.num_heads > cfg.num_heads_k
        and (cfg.use_local or (window_size[0] < 0 and window_size[1] < 0))
        and dropout_p == 0.0
        and alibi_slopes is None
    )
    if not should_hit_splitkv:
        return DenseCaseResult(
            cfg=cfg,
            head_dim=head_dim,
            status="FAIL",
            detail="splitkv_precondition_not_met",
        )

    softmax_scale = 1.0 / math.sqrt(head_dim)

    def run_kernel(num_splits: int) -> torch.Tensor:
        out = torch.zeros_like(q)
        ret = torch.ops._vllm_fa2_C.varlen_fwd(
            q,
            k,
            v,
            out,
            cu_seqlens_q,
            cu_seqlens_k,
            None,
            None,
            None,
            alibi_slopes,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            False,
            cfg.causal,
            window_size[0],
            window_size[1],
            0.0,
            False,
            num_splits,
            None,
        )
        if isinstance(ret, (list, tuple)) and len(ret) > 0:
            out = ret[0]
        return out

    out_base = run_kernel(base_num_splits)
    out_split = run_kernel(split_num_splits)

    torch.cuda.synchronize()

    if not torch.isfinite(out_base).all() or not torch.isfinite(out_split).all():
        return DenseCaseResult(
            cfg=cfg,
            head_dim=head_dim,
            status="FAIL",
            detail="kernel_output_has_nan_or_inf",
        )

    with torch.no_grad():
        ref_out = reference_varlen_attention(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            num_heads=cfg.num_heads,
            num_heads_k=cfg.num_heads_k,
            head_dim=head_dim,
            causal=cfg.causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
        )

    diff = torch.abs(out_split - out_base)
    diff_base_ref = torch.abs(out_base - ref_out)
    diff_split_ref = torch.abs(out_split - ref_out)
    max_diff = float(max(diff.max().item(), diff_base_ref.max().item(), diff_split_ref.max().item()))
    mean_diff = float(max(diff.mean().item(), diff_base_ref.mean().item(), diff_split_ref.mean().item()))

    pass_cond = (
        float(diff.max().item()) <= 5e-2
        and float(diff.mean().item()) <= 6e-3
        and float(diff_base_ref.max().item()) <= 8e-2
        and float(diff_base_ref.mean().item()) <= 8e-3
        and float(diff_split_ref.max().item()) <= 8e-2
        and float(diff_split_ref.mean().item()) <= 8e-3
    )

    return DenseCaseResult(
        cfg=cfg,
        head_dim=head_dim,
        status="PASS" if pass_cond else "FAIL",
        detail=f"split_vs_base_check(ns_base={base_num_splits},ns_split={split_num_splits})",
        max_diff=max_diff,
        mean_diff=mean_diff,
    )


def run_kblockm32_smoke_case(device: str, head_dim: int) -> SmokeCaseResult:
    sq = 16
    sk = 16
    num_heads = 8

    seed = 20260327 + head_dim
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    q = torch.randn(sq, num_heads, head_dim, dtype=torch.float16, device=device).contiguous()
    k = torch.randn(sk, num_heads, head_dim, dtype=torch.float16, device=device).contiguous()
    v = torch.randn(sk, num_heads, head_dim, dtype=torch.float16, device=device).contiguous()
    cu_seqlens_q = torch.tensor([0, sq], dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0, sk], dtype=torch.int32, device=device)
    out = torch.zeros_like(q)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    ret = torch.ops._vllm_fa2_C.varlen_fwd(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        None,
        None,
        None,
        None,
        sq,
        sk,
        0.0,
        softmax_scale,
        False,
        False,
        -1,
        -1,
        0.0,
        False,
        0,
        None,
    )

    if not isinstance(ret, (list, tuple)) or len(ret) < 2:
        return SmokeCaseResult(
            head_dim=head_dim,
            status="FAIL",
            detail="varlen_fwd_did_not_return_out_and_lse",
        )

    out = ret[0]
    lse = ret[1]
    if not torch.isfinite(out).all() or not torch.isfinite(lse).all():
        return SmokeCaseResult(
            head_dim=head_dim,
            status="FAIL",
            detail="kernel_output_or_lse_has_nan_or_inf",
        )

    with torch.no_grad():
        ref_out, ref_lse = reference_varlen_attention_and_lse(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            num_heads=num_heads,
            num_heads_k=num_heads,
            head_dim=head_dim,
            causal=False,
            window_size=(-1, -1),
            alibi_slopes=None,
        )

    out_diff = torch.abs(out - ref_out).float()
    lse_diff = torch.abs(lse - ref_lse).float()
    out_max_diff = float(out_diff.max().item())
    out_mean_diff = float(out_diff.mean().item())
    lse_max_diff = float(lse_diff.max().item())
    lse_mean_diff = float(lse_diff.mean().item())

    pass_cond = (
        out_max_diff <= 8e-2
        and out_mean_diff <= 8e-3
        and lse_max_diff <= 5e-3
        and lse_mean_diff <= 5e-4
    )
    return SmokeCaseResult(
        head_dim=head_dim,
        status="PASS" if pass_cond else "FAIL",
        detail="sq16_sk16_out_and_lse_reference_check",
        out_max_diff=out_max_diff,
        out_mean_diff=out_mean_diff,
        lse_max_diff=lse_max_diff,
        lse_mean_diff=lse_mean_diff,
    )


def run_kblockm32_smoke_suite(device: str, head_dims: List[int]) -> List[SmokeCaseResult]:
    return [run_kblockm32_smoke_case(device=device, head_dim=head_dim) for head_dim in head_dims]


def run_one_case(cfg: DenseCaseConfig, device: str, head_dim: int) -> DenseCaseResult:
    torch.manual_seed(20260307 + cfg.case_id)
    torch.cuda.manual_seed_all(20260307 + cfg.case_id)

    try:
        if cfg.suite == "numerical":
            return run_numerical_case(cfg, device=device, head_dim=head_dim)
        if cfg.suite == "splitkv":
            return run_splitkv_case(cfg, device=device, head_dim=head_dim)
        return DenseCaseResult(
            cfg=cfg,
            head_dim=head_dim,
            status="FAIL",
            detail=f"unknown_suite:{cfg.suite}",
        )
    except Exception as e:
        tb = traceback.format_exc(limit=6).replace("\n", " | ")
        return DenseCaseResult(
            cfg=cfg,
            head_dim=head_dim,
            status="FAIL",
            detail=f"exception: {str(e)} | tb: {tb}",
        )


def bool_mark(v: bool) -> str:
    return "Y" if v else "N"


def print_results_table(results: List[DenseCaseResult]) -> None:
    header = (
        "| id | head_dim | suite | H/Hk | Sq | Sk | nsplit | causal | dropout | local | alibi | varlen | status | max_diff | mean_diff | detail |"
    )
    sep = "|---:|---------:|:------:|:----:|---:|---:|------:|:------:|:-------:|:-----:|:-----:|:------:|:------:|--------:|---------:|:-------|"
    print("\n=== Dense 验证结果表 ===")
    print(header)
    print(sep)
    for r in results:
        md = "-" if r.max_diff is None else f"{r.max_diff:.6f}"
        me = "-" if r.mean_diff is None else f"{r.mean_diff:.6f}"
        c = r.cfg
        print(
            f"| {c.case_id} | {r.head_dim} | {c.suite} | {c.num_heads}/{c.num_heads_k} | {c.seqlen_q} | {c.seqlen_k} | {c.split_num_splits} | "
            f"{bool_mark(c.causal)} | {bool_mark(c.use_dropout)} | {bool_mark(c.use_local)} | {bool_mark(c.use_alibi)} | "
            f"{bool_mark(c.use_varlen)} | {r.status} | {md} | {me} | {r.detail} |"
        )


def print_summary(results: List[DenseCaseResult]) -> None:
    total = len(results)
    passed = sum(1 for x in results if x.status == "PASS")
    failed = total - passed

    print("\n=== Dense 汇总 ===")
    print(f"总用例数: {total}")
    print(f"通过: {passed}")
    print(f"失败: {failed}")

    by_suite: Dict[str, List[DenseCaseResult]] = {}
    for r in results:
        by_suite.setdefault(r.cfg.suite, []).append(r)

    for suite, suite_results in sorted(by_suite.items()):
        suite_total = len(suite_results)
        suite_pass = sum(1 for x in suite_results if x.status == "PASS")
        print(f"- suite={suite}: {suite_pass}/{suite_total} PASS")

    if failed > 0:
        print("\n=== 失败用例（仅列 id 与原因） ===")
        for r in results:
            if r.status == "FAIL":
                print(
                    f"- case_id={r.cfg.case_id}, head_dim={r.head_dim}, "
                    f"suite={r.cfg.suite}, detail={r.detail}"
                )


def print_smoke_results(results: List[SmokeCaseResult]) -> None:
    print("\n=== kBlockM=32 最小回归 ===")
    print("| head_dim | status | out_max | out_mean | lse_max | lse_mean | detail |")
    print("|---------:|:------:|--------:|---------:|--------:|---------:|:-------|")
    for r in results:
        out_max = "-" if r.out_max_diff is None else f"{r.out_max_diff:.6f}"
        out_mean = "-" if r.out_mean_diff is None else f"{r.out_mean_diff:.6f}"
        lse_max = "-" if r.lse_max_diff is None else f"{r.lse_max_diff:.6f}"
        lse_mean = "-" if r.lse_mean_diff is None else f"{r.lse_mean_diff:.6f}"
        print(f"| {r.head_dim} | {r.status} | {out_max} | {out_mean} | {lse_max} | {lse_mean} | {r.detail} |")

    passed = sum(1 for r in results if r.status == "PASS")
    print(f"smoke 通过: {passed}/{len(results)}")


def build_numerical_case_templates() -> List[Dict[str, object]]:
    seqlen_pairs = [
        # 16~128：小尺寸、非对齐、非对称
        (16, 16),
        (17, 31),
        (31, 17),
        (32, 48),
        (47, 79),
        (63, 65),
        (96, 80),
        (127, 129),
        # 160~1024：中小尺寸与边界附近
        (160, 96),
        (192, 256),
        (255, 257),
        (320, 384),
        (511, 513),
        (768, 640),
        (1024, 896),
        # 1k~4k：中尺度、复杂非对称
        (1536, 2048),
        (2048, 2048),
        (3072, 2816),
        (4096, 4096),
        (4095, 4103),
        # 4k~7k：大尺寸非对称与非对齐
        (5120, 4608),
        (6144, 5632),
        (7168, 7168),
        # 8k 量级基线与复杂组合
        (8192, 8192),
        (7936, 7936),
        (8192, 7680),
        (7680, 8192),
        (8128, 7872),
        (7872, 8128),
        (8191, 8203),
        (8053, 7901),
        (7901, 8053),
    ]

    templates: List[Dict[str, object]] = []
    for sq, sk in seqlen_pairs:
        for causal, use_local, use_alibi, use_varlen in product(
            [False, True],
            [False, True],
            [False, True],
            [False, True],
        ):
            templates.append(
                {
                    "suite": "numerical",
                    "seqlen_q": sq,
                    "seqlen_k": sk,
                    "num_heads": 8,
                    "num_heads_k": 8,
                    "split_num_splits": 0,
                    "causal": causal,
                    "use_dropout": False,
                    "use_local": use_local,
                    "use_alibi": use_alibi,
                    "use_varlen": use_varlen,
                }
            )
    return templates


def build_splitkv_case_templates() -> List[Dict[str, object]]:
    seqlen_pairs = [
        (1, 257),
        (1, 384),
        (1, 513),
        (1, 1024),
        (1, 1536),
    ]
    head_pairs = [
        (8, 2),
        (16, 4),
        (16, 2),
    ]
    split_choices = [2, 4]

    templates: List[Dict[str, object]] = []
    for sq, sk in seqlen_pairs:
        for num_heads, num_heads_k in head_pairs:
            for split_num_splits in split_choices:
                for use_varlen in [False, True]:
                    templates.append(
                        {
                            "suite": "splitkv",
                            "seqlen_q": sq,
                            "seqlen_k": sk,
                            "num_heads": num_heads,
                            "num_heads_k": num_heads_k,
                            "split_num_splits": split_num_splits,
                            "causal": False,
                            "use_dropout": False,
                            "use_local": False,
                            "use_alibi": False,
                            "use_varlen": use_varlen,
                        }
                    )
    templates.append(
        {
            "suite": "splitkv",
            "seqlen_q": 1,
            "seqlen_k": 257,
            "num_heads": 8,
            "num_heads_k": 2,
            "split_num_splits": 2,
            "causal": False,
            "use_dropout": False,
            "use_local": True,
            "use_alibi": False,
            "use_varlen": False,
        }
    )
    return templates


def build_case_matrix(dense_suite: str) -> List[DenseCaseConfig]:
    templates: List[Dict[str, object]] = []
    if dense_suite in ("all", "numerical"):
        templates.extend(build_numerical_case_templates())
    if dense_suite in ("all", "splitkv"):
        templates.extend(build_splitkv_case_templates())

    cases: List[DenseCaseConfig] = []
    for idx, t in enumerate(templates, start=1):
        cases.append(
            DenseCaseConfig(
                case_id=idx,
                suite=str(t["suite"]),
                seqlen_q=int(t["seqlen_q"]),
                seqlen_k=int(t["seqlen_k"]),
                num_heads=int(t["num_heads"]),
                num_heads_k=int(t["num_heads_k"]),
                split_num_splits=int(t["split_num_splits"]),
                causal=bool(t["causal"]),
                use_dropout=bool(t["use_dropout"]),
                use_local=bool(t["use_local"]),
                use_alibi=bool(t["use_alibi"]),
                use_varlen=bool(t["use_varlen"]),
            )
        )
    return cases


def filter_cases_by_case_ids(cases: List[DenseCaseConfig], case_ids: Optional[Set[int]]) -> List[DenseCaseConfig]:
    if case_ids is None:
        return cases
    return [c for c in cases if c.case_id in case_ids]


def get_selected_cases(dense_suite: Optional[str] = None) -> List[DenseCaseConfig]:
    resolved_dense_suite = get_dense_suite() if dense_suite is None else dense_suite
    cases = build_case_matrix(resolved_dense_suite)
    return filter_cases_by_case_ids(cases, get_case_id_filter())


def dense_case_pytest_id(cfg: DenseCaseConfig) -> str:
    return (
        f"{cfg.suite}-case{cfg.case_id}-sq{cfg.seqlen_q}-sk{cfg.seqlen_k}-"
        f"h{cfg.num_heads}hk{cfg.num_heads_k}-local{int(cfg.use_local)}-"
        f"alibi{int(cfg.use_alibi)}-causal{int(cfg.causal)}-varlen{int(cfg.use_varlen)}"
    )


def head_dim_pytest_id(head_dim: int) -> str:
    return f"hd{head_dim}"


def format_dense_failure(result: DenseCaseResult) -> str:
    return (
        f"case_id={result.cfg.case_id}, head_dim={result.head_dim}, suite={result.cfg.suite}, "
        f"status={result.status}, max_diff={result.max_diff}, mean_diff={result.mean_diff}, "
        f"detail={result.detail}"
    )


def format_smoke_failure(result: SmokeCaseResult) -> str:
    return (
        f"head_dim={result.head_dim}, status={result.status}, "
        f"out_max={result.out_max_diff}, out_mean={result.out_mean_diff}, "
        f"lse_max={result.lse_max_diff}, lse_mean={result.lse_mean_diff}, detail={result.detail}"
    )


def matrix_summary(cases: List[DenseCaseConfig], head_dims: List[int]) -> str:
    by_suite: Dict[str, int] = {}
    for c in cases:
        by_suite[c.suite] = by_suite.get(c.suite, 0) + 1

    suite_info = ", ".join(f"{k}:{v}" for k, v in sorted(by_suite.items()))
    hhk = sorted({(c.num_heads, c.num_heads_k) for c in cases})
    sk = sorted({c.seqlen_k for c in cases})
    ns = sorted({c.split_num_splits for c in cases if c.suite == "splitkv"})
    varlen = sorted({c.use_varlen for c in cases})
    return (
        f"head_dims={head_dims}, suite_count={{ {suite_info} }}, "
        f"H/Hk={hhk}, Sk={sk}, split_num_splits={ns}, varlen={varlen}"
    )


if pytest is not None:
    @pytest.fixture(scope="module", autouse=True)
    def _require_dense_runtime() -> None:
        ensure_runtime_available(for_pytest=True)


    def pytest_generate_tests(metafunc) -> None:
        if "head_dim" in metafunc.fixturenames:
            head_dims = get_selected_head_dims()
            metafunc.parametrize("head_dim", head_dims, ids=[head_dim_pytest_id(hd) for hd in head_dims])
        if "dense_case" in metafunc.fixturenames:
            cases = get_selected_cases()
            metafunc.parametrize("dense_case", cases, ids=[dense_case_pytest_id(case) for case in cases])


    def test_kblockm32_smoke(head_dim: int) -> None:
        result = run_kblockm32_smoke_case(device="cuda", head_dim=head_dim)
        assert result.status == "PASS", format_smoke_failure(result)


    def test_dense_matrix_case(dense_case: DenseCaseConfig, head_dim: int) -> None:
        result = run_one_case(dense_case, device="cuda", head_dim=head_dim)
        assert result.status == "PASS", format_dense_failure(result)


def main() -> int:
    try:
        ensure_runtime_available(for_pytest=False)
        dense_suite = get_dense_suite()
        head_dims = get_selected_head_dims()
        case_ids = get_case_id_filter()
    except (RuntimeError, ValueError) as exc:
        print(str(exc))
        return 1

    device = "cuda"
    name = torch.cuda.get_device_name()
    print(f"🖥️ 当前 GPU: {name}")
    print("✅ 成功导入 vllm_flash_attn")

    smoke_results = run_kblockm32_smoke_suite(device=device, head_dims=head_dims)
    print_smoke_results(smoke_results)

    cases = filter_cases_by_case_ids(build_case_matrix(dense_suite), case_ids)
    if case_ids is not None:
        print(f"按 CASE_IDS 过滤后保留 {len(cases)} 个用例: {sorted(case_ids)}")
        if not cases:
            print("CASE_IDS 过滤后没有可执行用例")
            return 1

    total_runs = len(cases) * len(head_dims)
    print(f"\n准备执行 {total_runs} 个 dense 场景用例")
    print("覆盖维度: numerical(通用 dense 数值对齐) + splitkv(decode split-kv 路径对齐)")
    print(f"矩阵详情: {matrix_summary(cases, head_dims)}")

    results: List[DenseCaseResult] = []
    run_idx = 0
    for head_dim in head_dims:
        print(f"\n--- head_dim={head_dim} ---")
        for cfg in cases:
            run_idx += 1
            print(
                f"[RUN {run_idx:04d}/{total_runs}][hd={head_dim}][{cfg.suite}] "
                f"H/Hk={cfg.num_heads}/{cfg.num_heads_k}, "
                f"Sq={cfg.seqlen_q}, Sk={cfg.seqlen_k}, "
                f"nsplit={cfg.split_num_splits}, "
                f"causal={cfg.causal}, dropout={cfg.use_dropout}, "
                f"local={cfg.use_local}, alibi={cfg.use_alibi}, varlen={cfg.use_varlen}"
            )
            result = run_one_case(
                cfg,
                device=device,
                head_dim=head_dim,
            )
            results.append(result)

    print_results_table(results)
    print_summary(results)

    all_pass = all(x.status == "PASS" for x in results) and all(x.status == "PASS" for x in smoke_results)
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
