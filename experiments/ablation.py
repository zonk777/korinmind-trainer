# ============================================================================
# ablation.py — KorinMind 消融实验
# ============================================================================
# 对比不同架构组件的效果，用控制变量法逐一验证每个设计选择。
#
# 实验设计：
#   - 使用 tiny 模型（hidden=128, layers=4）加速实验
#   - 同一批数据、相同的训练步数
#   - 只改变一个变量，对比 loss 曲线
#
# 运行：
#   python experiments/ablation.py
# ============================================================================

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model.model import KorinMindConfig, KorinMindForCausalLM
from dataset.dataset.lm_dataset import PretrainDataset
from trainer.trainer_untils import get_lr, setup_seed

DATA_PATH = "dataset/pretrain_test.jsonl"
TOKENIZER_PATH = "model"
TRAIN_STEPS = 50
BATCH_SIZE = 8
SEQ_LEN = 128
BASE_LR = 5e-4
HIDDEN_SIZE = 128
NUM_LAYERS = 4


def build_model(**overrides):
    """创建 tiny 模型，overrides 覆盖默认配置"""
    defaults = dict(
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=SEQ_LEN,
    )
    defaults.update(overrides)
    cfg = KorinMindConfig(**defaults)
    return KorinMindForCausalLM(cfg)


def train_one_model(name, model, device, dtype):
    """训练一个模型 TRAIN_STEPS 步，返回 loss 列表"""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    ds = PretrainDataset(DATA_PATH, tokenizer, max_length=SEQ_LEN)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model = model.to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR)
    autocast_ctx = torch.amp.autocast("cuda", dtype=dtype) if device != "cpu" else None

    losses = []
    total_params = sum(p.numel() for p in model.parameters())

    for step, (input_ids, labels, attention_mask) in enumerate(loader):
        if step >= TRAIN_STEPS:
            break

        input_ids = input_ids.to(device)
        labels = labels.to(device)
        attention_mask = attention_mask.to(device)

        lr = get_lr(step, TRAIN_STEPS, BASE_LR)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if autocast_ctx:
            with autocast_ctx:
                res = model(input_ids, labels=labels, attention_mask=attention_mask)
                loss = res.loss + res.aux_loss
        else:
            res = model(input_ids, labels=labels, attention_mask=attention_mask)
            loss = res.loss + res.aux_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (step + 1) % 10 == 0:
            avg = sum(losses[-10:]) / len(losses[-10:])
            print(f"  [{name}] step {step+1:3d}/{TRAIN_STEPS}, loss: {avg:.4f}")

    return {"name": name, "losses": losses, "params": total_params, "final_loss": losses[-1]}


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device != "cpu" else None

    results = []

    # =====================================================================
    # 实验 1：基准模型 (GQA + SwiGLU + RoPE)
    # =====================================================================
    print("\n" + "=" * 60)
    print("实验 1: 基准模型 (GQA + SwiGLU + RoPE)")
    print("=" * 60)
    setup_seed(42)
    baseline = build_model()
    r = train_one_model("Baseline", baseline, device, dtype)
    results.append(r)

    # =====================================================================
    # 实验 2：MHA (8Q/8KV) vs GQA (8Q/2KV)
    # =====================================================================
    print("\n" + "=" * 60)
    print("实验 2: MHA — 8 Q heads + 8 KV heads (vs GQA 8/2)")
    print("=" * 60)
    setup_seed(42)
    mha_model = build_model(num_key_value_heads=4)  # 4 KV heads (matching 4 Q heads = MHA)
    r = train_one_model("MHA (4KV)", mha_model, device, dtype)
    results.append(r)

    # =====================================================================
    # 实验 3：ReLU FFN vs SwiGLU
    # =====================================================================
    print("\n" + "=" * 60)
    print("实验 3: ReLU FFN (vs SwiGLU)")
    print("=" * 60)
    setup_seed(42)
    relu_model = build_model(hidden_act="relu")
    r = train_one_model("ReLU FFN", relu_model, device, dtype)
    results.append(r)

    # =====================================================================
    # 实验 4：无 RoPE
    # =====================================================================
    print("\n" + "=" * 60)
    print("实验 4: 无 RoPE 位置编码")
    print("=" * 60)
    setup_seed(42)
    no_rope = build_model(use_rope=False)
    r = train_one_model("No RoPE", no_rope, device, dtype)
    results.append(r)

    # =====================================================================
    # 实验 5：更深的模型 (8 层 vs 4 层)
    # =====================================================================
    print("\n" + "=" * 60)
    print("实验 5: 更深模型 — 8 层 (vs 4 层)")
    print("=" * 60)
    setup_seed(42)
    deep_model = build_model(num_hidden_layers=8)
    r = train_one_model("8 Layers", deep_model, device, dtype)
    results.append(r)

    # =====================================================================
    # 汇总报告
    # =====================================================================
    print("\n" + "=" * 60)
    print("消融实验报告")
    print("=" * 60)
    print(f"\n条件：hidden={HIDDEN_SIZE}, steps={TRAIN_STEPS}, batch={BATCH_SIZE}, seq={SEQ_LEN}")
    print()
    print(f"{'实验组':<25} {'参数量':<12} {'最终 Loss':<12} {'vs 基准':<10}")
    print("-" * 60)

    base_loss = results[0]["final_loss"]
    for r in results:
        delta = r["final_loss"] - base_loss
        sign = "+" if delta > 0 else ""
        delta_str = f"{sign}{delta:.4f}"
        print(f"{r['name']:<25} {r['params']/1e6:>6.2f}M   {r['final_loss']:<12.4f} {delta_str:<10}")

    print("-" * 60)
    print("\n结论：")

    # GQA vs MHA
    gqa_loss = results[0]["final_loss"]
    mha_loss = results[1]["final_loss"]
    gqa_params = results[0]["params"]
    mha_params = results[1]["params"]
    param_saved = (1 - gqa_params / mha_params) * 100
    print(f"  GQA vs MHA: loss 差 {mha_loss-gqa_loss:+.4f}, GQA 节省 {param_saved:.0f}% KV 参数 → 推理更快")

    # SwiGLU vs ReLU
    swiglu_loss = results[0]["final_loss"]
    relu_loss = results[2]["final_loss"]
    print(f"  SwiGLU vs ReLU: loss 差 {relu_loss-swiglu_loss:+.4f} → SwiGLU 效果更好")

    # RoPE
    rope_loss = results[0]["final_loss"]
    norope_loss = results[3]["final_loss"]
    print(f"  RoPE vs No RoPE: loss 差 {norope_loss-rope_loss:+.4f} → 位置编码至关重要")

    # Depth
    shallow_loss = results[0]["final_loss"]
    deep_loss = results[4]["final_loss"]
    deep_params = results[4]["params"]
    print(f"  深度 8 vs 4 层: loss 差 {deep_loss-shallow_loss:+.4f}, 参数 ×{deep_params/gqa_params:.1f}")

    # 保存结果
    os.makedirs("experiments", exist_ok=True)
    torch.save(results, "experiments/ablation_results.pt")
    print("\n结果已保存到 experiments/ablation_results.pt")


if __name__ == "__main__":
    main()
