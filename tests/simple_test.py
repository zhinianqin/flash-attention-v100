import math
import sys
from dataclasses import dataclass
from itertools import product
from typing import List, Optional, Tuple

import torch

# 尝试导入 vllm 编译好的算子库
try:
    import vllm_flash_attn  # noqa: F401
    print("✅ 成功导入 vllm_flash_attn")
except ImportError:
    print("❌ 无法导入 vllm_flash_attn，请检查是否已执行 pip install vllm-flash-attn")
    sys.exit(1)


@dataclass
class CaseConfig:
    case_id: int
    seqlen_q: int
    seqlen_k: int
    causal: bool
    use_dropout: bool
    use_local: bool
    use_alibi: bool
    use_varlen: bool


@dataclass
class CaseResult:
    cfg: CaseConfig
    status: str
    detail: str
    max_diff: Optional[float] = None
    mean_diff: Optional[float] = None


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
        k_b = k[k_start:k_end].float()  # [Sk, H, D]
        v_b = v[k_start:k_end].float()  # [Sk, H, D]

        sq = q_b.shape[0]
        sk = k_b.shape[0]

        if sq == 0:
            continue
        if sk == 0:
            out[q_start:q_end] = torch.zeros((sq, num_heads, head_dim), device=device, dtype=q.dtype)
            continue

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


def gen_lengths(base: int, use_varlen: bool) -> List[int]:
    if not use_varlen:
        return [base]
    return [base, max(1, base // 2), max(1, base - 7)]


def prepare_case_tensors(
    cfg: CaseConfig,
    device: str,
    num_heads: int,
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

    q = torch.randn(total_q, num_heads, head_dim, dtype=torch.float16, device=device).contiguous()
    k = torch.randn(total_k, num_heads, head_dim, dtype=torch.float16, device=device).contiguous()
    v = torch.randn(total_k, num_heads, head_dim, dtype=torch.float16, device=device).contiguous()

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
        # 按 batch 分配 slope，覆盖更一般的输入形态
        batch_size = len(q_lens)
        alibi_slopes = torch.rand(batch_size, num_heads, dtype=torch.float32, device=device) * 0.3
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


def run_one_case(cfg: CaseConfig, device: str, num_heads: int, head_dim: int) -> CaseResult:
    torch.manual_seed(20260307 + cfg.case_id)
    torch.cuda.manual_seed_all(20260307 + cfg.case_id)

    try:
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
        ) = prepare_case_tensors(cfg, device, num_heads, head_dim)

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
            return CaseResult(cfg=cfg, status="FAIL", detail="kernel_output_has_nan_or_inf")

        if dropout_p > 0.0:
            # 当前分支仅做前向可运行性校验（dropout mask 不对外暴露）
            return CaseResult(cfg=cfg, status="PASS", detail="dropout_run_ok_finite")

        with torch.no_grad():
            ref_out = reference_varlen_attention(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                num_heads=num_heads,
                head_dim=head_dim,
                causal=cfg.causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
            )

        diff = torch.abs(out - ref_out)
        max_diff = float(diff.max().item())
        mean_diff = float(diff.mean().item())

        # fp16 + FlashAttention 允许小幅数值误差
        pass_cond = (max_diff <= 8e-2) and (mean_diff <= 8e-3)

        return CaseResult(
            cfg=cfg,
            status="PASS" if pass_cond else "FAIL",
            detail="numerical_check",
            max_diff=max_diff,
            mean_diff=mean_diff,
        )
    except Exception as e:
        return CaseResult(cfg=cfg, status="FAIL", detail=f"exception: {str(e)}")


def bool_mark(v: bool) -> str:
    return "Y" if v else "N"


def print_results_table(results: List[CaseResult]) -> None:
    header = (
        "| id | Sq | Sk | causal | dropout | local | alibi | varlen | status | max_diff | mean_diff | detail |"
    )
    sep = "|---:|---:|---:|:------:|:-------:|:-----:|:-----:|:------:|:------:|--------:|---------:|:-------|"
    print("\n=== 验证结果表 ===")
    print(header)
    print(sep)
    for r in results:
        md = "-" if r.max_diff is None else f"{r.max_diff:.6f}"
        me = "-" if r.mean_diff is None else f"{r.mean_diff:.6f}"
        c = r.cfg
        print(
            f"| {c.case_id} | {c.seqlen_q} | {c.seqlen_k} | {bool_mark(c.causal)} | "
            f"{bool_mark(c.use_dropout)} | {bool_mark(c.use_local)} | {bool_mark(c.use_alibi)} | "
            f"{bool_mark(c.use_varlen)} | {r.status} | {md} | {me} | {r.detail} |"
        )


def print_summary(results: List[CaseResult]) -> None:
    total = len(results)
    passed = sum(1 for x in results if x.status == "PASS")
    failed = total - passed

    print("\n=== 汇总 ===")
    print(f"总用例数: {total}")
    print(f"通过: {passed}")
    print(f"失败: {failed}")

    if failed > 0:
        print("\n=== 失败用例（仅列 id 与原因） ===")
        for r in results:
            if r.status == "FAIL":
                print(f"- case_id={r.cfg.case_id}, detail={r.detail}")


def build_case_matrix() -> List[CaseConfig]:
    seqlen_pairs = [
        (16, 16),
        (32, 48),
        (64, 64),
        (96, 80),
    ]

    cases: List[CaseConfig] = []
    case_id = 1
    for sq, sk in seqlen_pairs:
        # 当前构建配置禁用 dropout，矩阵测试固定 dropout=0
        for causal, use_local, use_alibi, use_varlen in product(
            [False, True],
            [False, True],
            [False, True],
            [False, True],
        ):
            cases.append(
                CaseConfig(
                    case_id=case_id,
                    seqlen_q=sq,
                    seqlen_k=sk,
                    causal=causal,
                    use_dropout=False,
                    use_local=use_local,
                    use_alibi=use_alibi,
                    use_varlen=use_varlen,
                )
            )
            case_id += 1
    return cases


def main() -> int:
    if not torch.cuda.is_available():
        print("需要 GPU 环境")
        return 1

    device = "cuda"
    name = torch.cuda.get_device_name()
    print(f"🖥️ 当前 GPU: {name}")

    num_heads = 8
    head_dim = 64

    cases = build_case_matrix()
    print(f"\n准备执行 {len(cases)} 个场景用例")
    print("覆盖维度: seqlen_q/seqlen_k × causal × local × alibi × varlen（dropout 固定为 0）")

    results: List[CaseResult] = []
    for idx, cfg in enumerate(cases, start=1):
        print(
            f"[RUN {idx:03d}/{len(cases)}] "
            f"Sq={cfg.seqlen_q}, Sk={cfg.seqlen_k}, "
            f"causal={cfg.causal}, dropout={cfg.use_dropout}, "
            f"local={cfg.use_local}, alibi={cfg.use_alibi}, varlen={cfg.use_varlen}"
        )
        result = run_one_case(cfg, device=device, num_heads=num_heads, head_dim=head_dim)
        results.append(result)

    print_results_table(results)
    print_summary(results)

    all_pass = all(x.status == "PASS" for x in results)
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
