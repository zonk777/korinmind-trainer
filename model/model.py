# ============================================================================
# KorinMind — 从零实现的轻量级 Transformer 语言模型
# ============================================================================

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from transformers import GenerationMixin, PreTrainedModel, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast


class KorinMindConfig(PretrainedConfig):
    model_type = "korinmind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        use_rope: bool = True,  # 消融实验用：设为 False 可禁用 RoPE
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_rope = use_rope
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


# ---------------------------------------------------------------------------
# RMSNorm — 均方根归一化
# 比 LayerNorm 少算一个均值，速度快 10-15%，Llama / Qwen 系列都在用
# 公式：x * rsqrt(mean(x²) + eps)，再乘一个可学习的 weight
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # weight 是可学习的缩放参数，初始全为 1
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # rsqrt 是 1/sqrt，比 sqrt 再取倒数更快更稳定
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 先转 float32 做高精度计算，再转回原来的类型（bfloat16 等）
        return self.weight * self._norm(x.float()).type_as(x)


# ---------------------------------------------------------------------------
# precompute_freqs — 预计算 RoPE 旋转位置编码的 cos/sin 表格
#
# 支持 YaRN（Yet another RoPE extensioN）：
#   - 训练时用 2048 长度，推理时想扩展到 32768
#   - YaRN 对低频维度做插值缩放，高频维度保持不变，中间平滑过渡
#   - 这样不需要重新训练就能支持更长上下文
# ---------------------------------------------------------------------------
def precompute_freqs(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
):
    # 1. 标准 RoPE 频率：1 / (base^(2i/d))，i = 0,1,2,...,dim/2-1
    freqs, attn_factor = (
        1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)),
        1.0,
    )

    if rope_scaling is not None:
        # 2. 从配置中提取 YaRN 超参数
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 16)
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        attn_factor = rope_scaling.get("attention_factor", 1.0)

        # 只有当推理长度 > 训练长度时才缩放
        if end / orig_max > 1.0:
            # 3. 波长比例 → 维度索引 的映射函数
            def inv_dim(b):
                return (dim * math.log(orig_max / (b * 2 * math.pi))) / (
                    2 * math.log(rope_base)
                )

            # 4. 计算高频/低频的分界点
            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)

            # 5. 线性过渡 ramp：低维度（高频）= 0，高维度（低频）= 1，中间线性
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )

            # 6. 频率融合：f' = f * ((1-ramp) + ramp/factor)
            freqs = freqs * (1 - ramp + ramp / factor)

    # 7. 生成位置索引 [0, 1, 2, ..., end-1]
    t = torch.arange(end, device=freqs.device)

    # 8. 外积：每个位置 × 每个频率 → 旋转角度矩阵
    freqs = torch.outer(t, freqs).float()

    # 9. 取 cos 和 sin，拼成和 hidden_size 一样长的向量
    #    加上 attention_factor 补偿长距离下注意力被稀释的问题
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor

    return freqs_cos, freqs_sin


# ---------------------------------------------------------------------------
# apply_rotary_pos_emb — 把预计算好的 cos/sin 旋转应用到 Q 和 K 上
#
# 核心思想：把向量"旋转"一个角度，旋转角度和位置有关。
# 这样两个向量的内积就包含了它们之间的相对位置信息。
# ---------------------------------------------------------------------------
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # 把向量后半部分取负放到前面，前面放到后面，实现"旋转 90 度"
    def rotate_half(x):
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )

    # 复数乘法的实数形式：x*cos + rotate_half(x)*sin
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    )
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    )
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# repeat_kv — GQA（分组查询注意力）中的 KV 头复制
#
# 我们的模型有 8 个 Q 头但只有 2 个 KV 头。
# 每个 KV 头要给 4 个 Q 头提供信息，所以需要把 KV 复制 4 份。
# ---------------------------------------------------------------------------
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # 在维度上插入一个维度 → expand 复制 → reshape 成目标形状
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


# ============================================================================
# 第 3 步：Attention — 分组查询注意力（GQA）+ Flash Attention + KV Cache
# ============================================================================

class Attention(nn.Module):
    """
    分组查询注意力 (Grouped Query Attention)

    8 个 Q 头，2 个 KV 头 → 每 4 个 Q 头共享 1 组 KV
    这样推理时 KV Cache 只需要存 2 个头的 KV 而不是 8 个，显存节省 4 倍。

    流程图：
      x → Q, K, V 投影 → RoPE(Q,K) → repeat_kv(K,V) → QK^T/√d → softmax → ×V → O 投影 → 输出
    """

    def __init__(self, args: KorinMindConfig):
        super().__init__()

        # 如果没有指定 KV 头数，默认和 Q 头数一样（即普通注意力）
        self.num_key_value_heads = (
            args.num_attention_heads
            if args.num_key_value_heads is None
            else args.num_key_value_heads
        )

        # 确保 Q 头数是 KV 头数的整数倍
        assert args.num_attention_heads % self.num_key_value_heads == 0

        self.n_local_heads = args.num_attention_heads      # 8
        self.n_local_kv_heads = self.num_key_value_heads   # 2
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 4
        self.head_dim = args.hidden_size // args.num_attention_heads  # 512/8 = 64

        # Q 投影：hidden_size → 8heads × 64dim
        self.q_proj = nn.Linear(
            args.hidden_size, args.num_attention_heads * self.head_dim, bias=False
        )
        # K 投影：hidden_size → 2heads × 64dim (少！GQA 的关键)
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        # V 投影：hidden_size → 2heads × 64dim
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        # O 投影：8heads × 64dim → hidden_size
        self.o_proj = nn.Linear(
            args.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # 检查是否支持 Flash Attention（PyTorch 2.0+ 内置）
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and args.flash_attention
        )

        # 消融实验：是否使用 RoPE
        self.use_rope = getattr(args, "use_rope", True)

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        bsz, seq_len, _ = x.shape

        # 1. 线性投影：X → Q, K, V
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # 2. 改变形状：(batch, seq, hidden) → (batch, seq, n_heads, head_dim)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # 3. 施加 RoPE 旋转位置编码
        cos, sin = position_embeddings
        if self.use_rope:
            xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # 4. KV Cache：推理时拼接过去的 K/V，避免重复计算
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # 5. 转换为标准注意力格式：(batch, n_heads, seq, head_dim)
        #    KV 头先通过 repeat_kv 从 2 复制到 8
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )

        # 6. 计算注意力
        if (
            self.flash
            and (seq_len > 1)
            and (past_key_value is None)
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            # Flash Attention 路径：更快、更省显存
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # 手动实现路径（有 KV cache 或需要自定义 mask 时走这里）
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # 因果遮罩：当前位置看不到未来的内容
            scores[:, :, :, -seq_len:] += torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1,
            )

            # 额外的 attention mask（比如 padding 遮罩）
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        # 7. 合并多头：(batch, n_heads, seq, head_dim) → (batch, seq, hidden)
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)

        # 8. 最终输出投影 + dropout
        output = self.resid_dropout(self.o_proj(output))

        return output, past_kv


# ============================================================================
# 第 4 步：前馈网络 — FeedForward (SwiGLU) + MoEGate + MoEFeedForward
# ============================================================================

# ---------------------------------------------------------------------------
# FeedForward — SwiGLU 前馈网络
#
# 公式：SiLU(gate_proj(x)) * up_proj(x) → down_proj(...)
# 普通 FFN 只用一个矩阵，SwiGLU 用了三个（gate/up/down），
# 多了一个"门控"机制，让模型自己决定哪些信息该通过。
# 中间层大小 = hidden_size * 8/3，对齐到 64 的倍数
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, config: KorinMindConfig):
        super().__init__()

        # 自动计算中间层大小：512 * 8/3 ≈ 1365 → 对齐到 64 的倍数 → 1408
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        # 三个投影矩阵
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )

        self.dropout = nn.Dropout(config.dropout)
        # SiLU 激活函数：x * sigmoid(x)，比 ReLU 更平滑
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # SwiGLU 的核心：SiLU(gate(x)) * up(x) → 门控信号 × 信息信号
        gated = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(gated))


# ---------------------------------------------------------------------------
# MoEGate — 混合专家路由器
#
# 决定每个 token 应该交给哪几个专家处理（Top-K 选择）。
# 还要计算"负载均衡损失"——防止所有 token 都选同一个专家，其他专家闲置。
# ---------------------------------------------------------------------------
class MoEGate(nn.Module):
    def __init__(self, config: KorinMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok        # 每 token 选几个专家（默认 2）
        self.n_routed_experts = config.n_routed_experts  # 总共有几个路由专家（默认 4）

        self.scoring_func = config.scoring_func          # 评分方式：softmax
        self.alpha = config.aux_loss_alpha               # 辅助损失系数
        self.seq_aux = config.seq_aux                     # 是否用序列级辅助损失

        self.norm_topk_prob = config.norm_topk_prob       # 是否归一化 top-k 权重
        self.gating_dim = config.hidden_size

        # 路由权重矩阵：(n_routed_experts, hidden_size)
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape

        # 展平 batch 和序列维度：(B*L, hidden)
        hidden_states = hidden_states.view(-1, h)

        # 计算每个 token 对每个专家的分数
        logits = F.linear(hidden_states, self.weight, None)

        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        # Top-K 选择：取分数最高的 K 个专家
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 归一化 top-k 权重（让 K 个专家的权重加起来 = 1）
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # 计算辅助损失（只在训练时）
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)

            if self.seq_aux:
                # 序列级辅助损失：每条样本单独计算
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(
                    bsz, self.n_routed_experts, device=hidden_states.device
                )
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                    dim=1
                ).mean() * self.alpha
            else:
                # 全局辅助损失
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = scores.new_zeros(1).squeeze()

        return topk_idx, topk_weight, aux_loss


# ---------------------------------------------------------------------------
# MoEFeedForward — 混合专家前馈网络
#
# 包含：n_routed_experts 个路由专家 + n_shared_experts 个共享专家
# - 路由专家：每个 token 只走 top-k 个（稀疏激活），省计算量
# - 共享专家：所有 token 都走，提供通用知识
#
# 训练路径：重复输入 → 每个专家处理自己分配到的 token → 加权求和
# 推理路径：按专家分组批量处理（moe_infer），避免遍历每个 token
# ---------------------------------------------------------------------------
class MoEFeedForward(nn.Module):
    def __init__(self, config: KorinMindConfig):
        super().__init__()
        self.config = config

        # 路由专家列表
        self.experts = nn.ModuleList(
            [FeedForward(config) for _ in range(config.n_routed_experts)]
        )

        # 路由器
        self.gate = MoEGate(config)

        # 共享专家（全部 token 都经过）
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [FeedForward(config) for _ in range(config.n_shared_experts)]
            )

    def forward(self, x):
        identity = x
        orig_shape = x.shape

        # 1. 路由器决定每个 token 选哪些专家
        topk_idx, topk_weight, aux_loss = self.gate(x)

        # 2. 展平：(B, L, hidden) → (B*L, hidden)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        if self.training:
            # ---- 训练路径 ----
            # 每个 token 复制 K 份（K = num_experts_per_tok），分别发给 K 个专家
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=x.dtype)

            for i, expert in enumerate(self.experts):
                # 找到分配给专家 i 的 token，只处理这些
                expert_out = expert(x[flat_topk_idx == i])
                if expert_out.shape[0] > 0:
                    y[flat_topk_idx == i] = expert_out.to(y.dtype)
                else:
                    # 保持计算图连通（DDP 要求每个专家都参与）
                    y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(
                        p.sum() for p in expert.parameters()
                    )

            # 加权求和：每个 token 的 K 个专家输出 × 各自权重 → 加起来
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            # ---- 推理路径 ----
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(
                *orig_shape
            )

        # 3. 加上共享专家的输出（所有 token 都过一遍）
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)

        # 把辅助损失挂在 self 上，外层模型会收集
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """推理时的批量专家计算：按专家分组，同一专家的 token 一次性处理"""
        expert_cache = torch.zeros_like(x)

        # 按专家索引排序，把同一专家的 token 聚在一起
        idxs = flat_expert_indices.argsort()

        # 每个专家被分配了多少 token（累积和）
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)

        # 映射回原始 token 索引
        token_idxs = idxs // self.config.num_experts_per_tok

        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue

            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]

            # 专家一次性处理这批 token
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])

            # scatter_add 把结果放回对应位置
            expert_cache.scatter_add_(
                0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out
            )

        return expert_cache


# ============================================================================
# 第 5 步：组装完整模型 — Block → Model → CausalLM
# ============================================================================

# ---------------------------------------------------------------------------
# KorinMindBlock — 一个 Transformer 层（Pre-Norm 架构）
#
# 流程：x → RMSNorm → Attention → +残差 → RMSNorm → FFN/MoE → +残差
# "Pre-Norm"意思是先归一化再做计算（而不是先计算再归一化），
# 这是 Llama 系列的标准做法，训练更稳定。
# ---------------------------------------------------------------------------
class KorinMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: KorinMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.self_attention = Attention(config)

        self.layer_id = layer_id

        # 两个 RMSNorm：一个在 Attention 前，一个在 FFN 前
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # 根据配置选择普通 FFN 或 MoE FFN
        self.mlp = (
            FeedForward(config)
            if not config.use_moe
            else MoEFeedForward(config)
        )

    def forward(
        self,
        hidden_states,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # 残差连接
        res = hidden_states

        # 1. Pre-Norm Attention
        hidden_states, present_key_value = self.self_attention(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )

        # 2. 残差连接
        hidden_states = res + hidden_states

        # 3. Pre-Norm FFN/MoE + 残差连接
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )

        return hidden_states, present_key_value


# ---------------------------------------------------------------------------
# KorinMindModel — Transformer 主干
#
# 职责：
#   1. 把 input_ids 通过 embedding 变成向量
#   2. 预计算 RoPE 的 cos/sin 表（注册为 buffer，随模型保存/加载）
#   3. 逐层传递 hidden_states，收集 KV cache 和 MoE 辅助损失
#   4. 最后过一层 RMSNorm
# ---------------------------------------------------------------------------
class KorinMindModel(nn.Module):
    def __init__(self, config: KorinMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

        # Token 嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        # 堆叠 N 层 Transformer Block
        self.layers = nn.ModuleList(
            [KorinMindBlock(i, config) for i in range(self.num_hidden_layers)]
        )

        # 最终归一化
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 预计算 RoPE cos/sin 并注册为 buffer
        # persistent=False → 不保存到 state_dict（可以推理时重新算）
        freqs_cos, freqs_sin = precompute_freqs(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        batch_size, seq_length = input_ids.shape

        # 兼容 HuggingFace DynamicCache
        if hasattr(past_key_values, "layers"):
            if hasattr(past_key_values, "to_legacy_cache"):
                past_key_values = past_key_values.to_legacy_cache()
            else:
                past_key_values = None

        past_key_values = past_key_values or [None] * len(self.layers)

        # 计算当前位置偏移（用于 RoPE）
        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )

        # Token Embedding + Dropout
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 切片出当前位置对应的 RoPE cos/sin
        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_length],
            self.freqs_sin[start_pos : start_pos + seq_length],
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(
            zip(self.layers, past_key_values)
        ):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)

        # 最终归一化
        hidden_states = self.norm(hidden_states)

        # 汇总所有 MoE 层的辅助损失
        aux_loss = sum(
            [
                layer.mlp.aux_loss
                for layer in self.layers
                if isinstance(layer.mlp, MoEFeedForward)
            ],
            hidden_states.new_zeros(1).squeeze(),
        )

        return hidden_states, presents, aux_loss


# ---------------------------------------------------------------------------
# KorinMindForCausalLM — 最终模型（带 LM Head）
#
# 继承 PreTrainedModel + GenerationMixin 获得 HuggingFace 生态能力：
#   - model.generate() 自动生成文本
#   - model.save_pretrained() / model.from_pretrained() 保存/加载
#   - AutoModel 自动识别
#
# 关键设计：权重绑定（Weight Tying）
#   embed_tokens.weight = lm_head.weight
#   输入嵌入和输出投影共享参数，节省 hidden × vocab = 512×6400 ≈ 3.3M 参数
# ---------------------------------------------------------------------------
class KorinMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = KorinMindConfig

    def __init__(self, config: KorinMindConfig):
        super().__init__(config)
        self.model = KorinMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 权重绑定：embedding 和 lm_head 共享权重
        self.model.embed_tokens.weight = self.lm_head.weight

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """束搜索时重排 KV cache（必须保留的接口）"""
        reordered_past = []
        for layer_past in past_key_values:
            reordered_past.append(
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                )
            )
        return reordered_past

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **args,
    ):
        # 1. 过 Transformer 主干
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args,
        )

        # 2. 只保留最后 logits_to_keep 个位置的 logits（推理优化）
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # 3. 计算交叉熵损失（训练时）
        loss = None
        if labels is not None:
            # Shift：预测第 t 个 token 应该用的 label 是第 t 个 token
            # logits[..., :-1, :] 对应 predict(t), labels[..., 1:] 对应 answer(t)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,  # padding 位置不参与 loss
            )

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )
        output.aux_loss = aux_loss
        return output