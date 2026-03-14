import math
import os
import sys
import traceback
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Tuple

import torch

# 避免仓库内源码包(vllm_flash_attn/)覆盖已安装wheel，导致找不到编译后的 .so
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path = [
    p for p in sys.path
    if os.path.abspath(p if p else os.getcwd()) != _REPO_ROOT
]

# 尝试导入 vllm 编译好的算子库
try:
    import vllm_flash_attn  # noqa: F401
    print("✅ 成功导入 vllm_flash_attn")
except ImportError:
    print("❌ 无法导入 vllm_flash_attn，请检查是否已执行 pip install vllm-flash-attn")
    sys.exit(1)


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
        return DenseCaseResult(cfg=cfg, status="FAIL", detail="kernel_output_has_nan_or_inf")

    if dropout_p > 0.0:
        return DenseCaseResult(cfg=cfg, status="PASS", detail="dropout_run_ok_finite")

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
        and window_size[0] < 0
        and window_size[1] < 0
        and dropout_p == 0.0
        and alibi_slopes is None
    )
    if not should_hit_splitkv:
        return DenseCaseResult(
            cfg=cfg,
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
        return DenseCaseResult(cfg=cfg, status="FAIL", detail="kernel_output_has_nan_or_inf")

    diff = torch.abs(out_split - out_base)
    max_diff = float(diff.max().item())
    mean_diff = float(diff.mean().item())

    pass_cond = (max_diff <= 5e-2) and (mean_diff <= 6e-3)

    return DenseCaseResult(
        cfg=cfg,
        status="PASS" if pass_cond else "FAIL",
        detail=f"split_vs_base_check(ns_base={base_num_splits},ns_split={split_num_splits})",
        max_diff=max_diff,
        mean_diff=mean_diff,
    )


def run_one_case(cfg: DenseCaseConfig, device: str, head_dim: int) -> DenseCaseResult:
    torch.manual_seed(20260307 + cfg.case_id)
    torch.cuda.manual_seed_all(20260307 + cfg.case_id)

    try:
        if cfg.suite == "numerical":
            return run_numerical_case(cfg, device=device, head_dim=head_dim)
        if cfg.suite == "splitkv":
            return run_splitkv_case(cfg, device=device, head_dim=head_dim)
        return DenseCaseResult(cfg=cfg, status="FAIL", detail=f"unknown_suite:{cfg.suite}")
    except Exception as e:
        tb = traceback.format_exc(limit=6).replace("\n", " | ")
        return DenseCaseResult(cfg=cfg, status="FAIL", detail=f"exception: {str(e)} | tb: {tb}")


def bool_mark(v: bool) -> str:
    return "Y" if v else "N"


def print_results_table(results: List[DenseCaseResult]) -> None:
    header = (
        "| id | suite | H/Hk | Sq | Sk | nsplit | causal | dropout | local | alibi | varlen | status | max_diff | mean_diff | detail |"
    )
    sep = "|---:|:------:|:----:|---:|---:|------:|:------:|:-------:|:-----:|:-----:|:------:|:------:|--------:|---------:|:-------|"
    print("\n=== Dense 验证结果表 ===")
    print(header)
    print(sep)
    for r in results:
        md = "-" if r.max_diff is None else f"{r.max_diff:.6f}"
        me = "-" if r.mean_diff is None else f"{r.mean_diff:.6f}"
        c = r.cfg
        print(
            f"| {c.case_id} | {c.suite} | {c.num_heads}/{c.num_heads_k} | {c.seqlen_q} | {c.seqlen_k} | {c.split_num_splits} | "
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
                print(f"- case_id={r.cfg.case_id}, suite={r.cfg.suite}, detail={r.detail}")


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


def matrix_summary(cases: List[DenseCaseConfig]) -> str:
    by_suite: Dict[str, int] = {}
    for c in cases:
        by_suite[c.suite] = by_suite.get(c.suite, 0) + 1

    suite_info = ", ".join(f"{k}:{v}" for k, v in sorted(by_suite.items()))
    hhk = sorted({(c.num_heads, c.num_heads_k) for c in cases})
    sk = sorted({c.seqlen_k for c in cases})
    ns = sorted({c.split_num_splits for c in cases if c.suite == "splitkv"})
    varlen = sorted({c.use_varlen for c in cases})
    return (
        f"suite_count={{ {suite_info} }}, H/Hk={hhk}, Sk={sk}, split_num_splits={ns}, varlen={varlen}"
    )


def main() -> int:
    if not torch.cuda.is_available():
        print("需要 GPU 环境")
        return 1

    dense_suite = os.environ.get("DENSE_SUITE", "all").strip().lower()
    if dense_suite not in {"all", "numerical", "splitkv"}:
        print("DENSE_SUITE 仅支持: all | numerical | splitkv")
        return 1

    device = "cuda"
    name = torch.cuda.get_device_name()
    print(f"🖥️ 当前 GPU: {name}")

    head_dim = 64

    cases = build_case_matrix(dense_suite)
    case_ids_env = os.environ.get("CASE_IDS", "").strip()
    if case_ids_env:
        wanted = {int(x) for x in case_ids_env.split(",") if x.strip()}
        cases = [c for c in cases if c.case_id in wanted]
        print(f"按 CASE_IDS 过滤后保留 {len(cases)} 个用例: {sorted(wanted)}")
        if not cases:
            print("CASE_IDS 过滤后没有可执行用例")
            return 1

    print(f"\n准备执行 {len(cases)} 个 dense 场景用例")
    print("覆盖维度: numerical(通用 dense 数值对齐) + splitkv(decode split-kv 路径对齐)")
    print(f"矩阵详情: {matrix_summary(cases)}")

    results: List[DenseCaseResult] = []
    for idx, cfg in enumerate(cases, start=1):
        print(
            f"[RUN {idx:03d}/{len(cases)}][{cfg.suite}] "
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

    all_pass = all(x.status == "PASS" for x in results)
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
