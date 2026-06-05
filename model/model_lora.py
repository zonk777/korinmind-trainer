# ============================================================================
# model_lora.py — 从零实现 LoRA（低秩适配）
# ============================================================================
# LoRA 核心思想：
#   冻结原始权重 W，在旁边加两个小矩阵 A 和 B
#   前向传播变成：output = W·x + (α/r)·B·A·x
#
#   参数量对比（以 hidden_size=512, rank=8 为例）：
#     原始 Linear: 512×512 = 262K 参数
#     LoRA A+B:   512×8 + 8×512 = 8K 参数 → 只有原来的 3%
#
# 优势：
#   - 训练快（只更新 1-5% 的参数）
#   - 显存省（冻结的权重不需要存梯度）
#   - 可插拔（保存多组 LoRA 权重，切换不同能力）
# ============================================================================

import torch
from torch import nn


class LoRA(nn.Module):
    """
    低秩适配器：ΔW = B·A

    Parameters
    ----------
    in_features : int   — 输入维度
    out_features : int  — 输出维度
    rank : int          — 低秩矩阵的秩（r << min(in, out)）
    alpha : float       — 缩放系数，实际输出 = (α/r) · B·A·x
    """

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float = 16.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # 缩放因子，控制 LoRA 的影响力

        # A: 降维矩阵 (in → rank)，高斯初始化
        self.A = nn.Linear(in_features, rank, bias=False)
        # B: 升维矩阵 (rank → out)，全零初始化
        self.B = nn.Linear(rank, out_features, bias=False)

        # B 初始化为零确保训练开始时 ΔW = 0，不破坏预训练权重
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.scaling * self.B(self.A(x))


# ============================================================================
# apply_lora — 将 LoRA 注入模型的所有方阵 Linear 层
# ============================================================================
# 只对 in_features == out_features 的 Linear 层加 LoRA（通常是 Q/K/V/O 投影），
# 跳过 FFN 中 in≠out 的层（gate/up 是 512→1408，不符合条件）。
# ============================================================================

def apply_lora(model: nn.Module, rank: int = 8, alpha: float = 16.0):
    """为模型中所有方阵 Linear 层注入 LoRA 适配器"""
    device = next(model.parameters()).device

    for name, module in model.named_modules():
        # 只对方阵 Linear 层加 LoRA（in == out 才是 Q/K/V/O 投影）
        if (
            isinstance(module, nn.Linear)
            and module.weight.shape[0] == module.weight.shape[1]
        ):
            lora = LoRA(
                module.weight.shape[0], module.weight.shape[1],
                rank=rank, alpha=alpha,
            ).to(device)

            # 把 LoRA 模块挂到原 Linear 层上
            setattr(module, "lora", lora)

            # 保存原始 forward，用新 forward 替换
            original_forward = module.forward
            module._original_forward = original_forward

            def make_forward(orig_forward, lora_module):
                def forward_with_lora(x):
                    return orig_forward(x) + lora_module(x)
                return forward_with_lora

            module.forward = make_forward(original_forward, lora)

    # 冻结所有非 LoRA 参数
    _freeze_except_lora(model)


def _freeze_except_lora(model: nn.Module):
    """冻结所有参数，只保留 LoRA 参数可训练"""
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


# ============================================================================
# save_lora / load_lora — 只保存/加载 LoRA 权重
# ============================================================================

def save_lora(model: nn.Module, path: str):
    """只保存 LoRA 权重（不含冻结的原始模型权重）"""
    raw_model = getattr(model, "_orig_mod", model)  # 兼容 torch.compile

    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, "lora"):
            # 去掉 module. 前缀（兼容 DDP）
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {
                f"{clean_name}.lora.{k}": v
                for k, v in module.lora.state_dict().items()
            }
            state_dict.update(lora_state)

    torch.save(state_dict, path)
    print(f"LoRA weights saved to {path} ({len(state_dict)} tensors)")


def load_lora(model: nn.Module, path: str):
    """加载 LoRA 权重到模型中"""
    device = next(model.parameters()).device
    state_dict = torch.load(path, map_location=device, weights_only=True)

    # 去掉 DDP 的 module. 前缀
    state_dict = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }

    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            lora_state = {
                k.replace(f"{name}.lora.", ""): v
                for k, v in state_dict.items()
                if f"{name}.lora." in k
            }
            if lora_state:
                module.lora.load_state_dict(lora_state)

    print(f"LoRA weights loaded from {path}")


# ============================================================================
# merge_lora — 将 LoRA 权重合并到原始模型中
# ============================================================================
# 合并后模型变为普通模型，推理时没有 LoRA 额外开销。
# 合并公式：W_merged = W + (α/r) · B·A
# ============================================================================

def merge_lora(model: nn.Module):
    """将 LoRA 权重合并到原始 Linear 层中，并移除 LoRA 模块"""
    raw_model = getattr(model, "_orig_mod", model)

    for name, module in raw_model.named_modules():
        if hasattr(module, "lora"):
            lora = module.lora
            # W_merged = W + scaling · B @ A
            delta_w = lora.scaling * (lora.B.weight @ lora.A.weight)
            module.weight.data = module.weight.data + delta_w

            # 恢复原始 forward
            original_forward = getattr(module, "_original_forward", None)
            if original_forward is not None:
                module.forward = original_forward

            del module.lora

    print("LoRA weights merged into base model")
