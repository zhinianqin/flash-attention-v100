import torch
import math
import sys

# 尝试导入 vllm 编译好的算子库
try:
    import vllm_flash_attn
    print("✅ 成功导入 vllm_flash_attn")
except ImportError:
    print("❌ 无法导入 vllm_flash_attn，请检查是否已执行 pip install vllm-flash-attn")
    sys.exit(1)

def run_precision_test(seqlen_q=32, seqlen_k=32, num_heads=8, head_dim=64):
    device = "cuda"
    dtype = torch.float16
    
    # 1. 初始化输入 (确保连续)
    q = torch.randn(seqlen_q, num_heads, head_dim, dtype=dtype, device=device).contiguous()
    k = torch.randn(seqlen_k, num_heads, head_dim, dtype=dtype, device=device).contiguous()
    v = torch.randn(seqlen_k, num_heads, head_dim, dtype=dtype, device=device).contiguous()


    # 打印第 5 个 token，第 2 个 Head 的所有维度
    # print("第 5 行, 第 2 个 Head 的数据:")
    # print(q[4, 1, :]) # 验证通过
    # print(k[4, 1, :]) # 验证通过

    # 初始化输出为 0，防止 nan 干扰判断
    out = torch.zeros_like(q) 
    softmax_scale = 1.0 / math.sqrt(head_dim)

    cu_seqlens_q = torch.tensor([0, seqlen_q], dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0, seqlen_k], dtype=torch.int32, device=device)

    # 2. 调用 Kernel (带上之前修正的 num_splits=0)
    try:
        torch.ops._vllm_fa2_C.varlen_fwd(
            q, k, v, out,
            cu_seqlens_q, cu_seqlens_k,
            None, None, None, None,
            seqlen_q, seqlen_k,
            0.0, softmax_scale, False, False,
            -1, -1, 0.0, False, 
            0,    # num_splits
            None  # gen
        )
        torch.cuda.synchronize()
    except Exception as e:
        print(f"💥 Kernel 运行失败: {e}")
        return

    # 3. 参考值计算
    with torch.no_grad():
        # (Total, H, D) -> (B, H, S, D)
        q_ref = q.view(1, seqlen_q, num_heads, head_dim).transpose(1, 2).float()
        k_ref = k.view(1, seqlen_k, num_heads, head_dim).transpose(1, 2).float()
        v_ref = v.view(1, seqlen_k, num_heads, head_dim).transpose(1, 2).float()

        attn = torch.matmul(q_ref, k_ref.transpose(-2, -1)) * softmax_scale

        attn = torch.softmax(attn, dim=-1)
        ref_out = torch.matmul(attn, v_ref).half()
        
        # 这里的 reshape 是安全的，transpose 后的 reshape 会自动处理内存
        ref_out = ref_out.transpose(1, 2).reshape(seqlen_q, num_heads, head_dim)

    # 4. 健壮的精度比对
    # 使用 torch.isnan 检查是否存在非法值
    if torch.isnan(out).any():
        print("❌ 警告: Kernel 输出包含 NaN!")
    
    diff = torch.abs(out - ref_out)
    max_diff = torch.max(diff).item()
    
    if max_diff < 1e-3:
        print(f"✅ 验证通过! Max Diff: {max_diff:.6f}")
    else:
        print(f"❌ 精度不合格! Max Diff: {max_diff:.6f}")
        # 使用 flatten() 或 reshape(-1) 避开 view 的 stride 问题
        print(f"Kernel sample: {out.flatten()[:3].tolist()}")
        print(f"Ref sample:    {ref_out.flatten()[:3].tolist()}")
        
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("需要 GPU 环境")
        exit()

    # 检查 GPU 型号
    name = torch.cuda.get_device_name()
    print(f"🖥️ 当前 GPU: {name}")

    # 测试几种常见的 head_dim
    for d in [32, 64, 96, 128, 192, 256]:
        run_precision_test(head_dim=d)
    
    print("\n✨ 所有测试完成")