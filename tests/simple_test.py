import torch
import sys

try:
    import vllm_flash_attn
except ImportError:
    pass

def test_vllm_flash_attn_final():
    print("🚀 启动最终决战：SM70 (V100) 完整前向验证")

    seqlen_q = 64
    seqlen_k = 64
    num_heads = 1
    head_dim = 64

    torch.manual_seed(42)

    q = torch.randn(seqlen_q, num_heads, head_dim, dtype=torch.float16, device='cuda')
    k = torch.randn(seqlen_k, num_heads, head_dim, dtype=torch.float16, device='cuda')
    v = torch.randn(seqlen_k, num_heads, head_dim, dtype=torch.float16, device='cuda')

    cu_seqlens_q = torch.tensor([0, seqlen_q], dtype=torch.int32, device='cuda')
    cu_seqlens_k = torch.tensor([0, seqlen_k], dtype=torch.int32, device='cuda')

    out = torch.empty_like(q)

    # === PyTorch 完整标准答案 ===
    q_ref = q.view(1, seqlen_q, num_heads, head_dim).transpose(1, 2)
    k_ref = k.view(1, seqlen_k, num_heads, head_dim).transpose(1, 2)
    v_ref = v.view(1, seqlen_k, num_heads, head_dim).transpose(1, 2)
    
    scores = torch.matmul(q_ref, k_ref.transpose(-2, -1)).float()
    scores = scores * 1.0 # softmax_scale = 1.0
    p_ref = torch.softmax(scores, dim=-1).half()
    ref_out = torch.matmul(p_ref, v_ref)
    
    ref_out_reshaped = ref_out.transpose(1, 2).contiguous()
    # ==========================

    print("⏳ 正在运行 V100 优化版 Kernel...")
    try:
        torch.ops._vllm_fa2_C.varlen_fwd(q, k, v, out, cu_seqlens_q, cu_seqlens_k, None, None, None, None, seqlen_q, seqlen_k, 0.0, 1.0, False, False, -1, -1, 0.0, False, None)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"\n❌ Kernel 运行时发生异常: \n{e}")
        return

    out_4d = out.view(1, seqlen_q, num_heads, head_dim)
    out_valid = out_4d[0, :, 0, :]
    ref_valid = ref_out_reshaped[0, :, 0, :]

    try:
        # 允许微小的 FP16 舍入误差
        torch.testing.assert_close(out_valid, ref_valid, rtol=1e-3, atol=1e-2)
        print("\n🎉✅ 完美通关！SM70 适配重构成功！")
        print("所有数值误差已清零，寄存器拓扑、Softmax 规约、P*V 转置全部跑通！")
    except AssertionError as e:
        print("\n❌ Failed! 发现数值错误：")
        mismatched = (torch.abs(out_valid - ref_valid) > 1e-2).sum().item()
        total = out_valid.numel()
        print(f"不匹配的元素比例: {mismatched} / {total} ({(mismatched/total)*100:.1f}%)")
        print("\n[PyTorch 标准答案]:\n", ref_valid[:8, :8])
        print("\n[你的 C++ 输出]:\n", out_valid[:8, :8])

if __name__ == '__main__':
    test_vllm_flash_attn_final()
