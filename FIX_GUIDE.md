# KorinMind 修改指南

基于 2026-06-05 深度评估报告，所有修改建议按优先级排列，包含具体代码修改方案。

---

## 🔴 P0 — 必须立即修复（影响功能正确性）

### P0-1: 修复 `merge_lora` 无法恢复原始 forward

**文件**: `model/model_lora.py`
**问题**: `apply_lora` 替换了 `module.forward` 但从未保存原始 forward；`merge_lora` 尝试读取 `_original_forward` 但永远是 None，导致 merge 后模型调用已删除的 lora 模块而崩溃。

**修改**:

```python
# model/model_lora.py, apply_lora 函数中,第 79-86 行

# ===== 修改前 =====
def make_forward(orig_forward, lora_module):
    def forward_with_lora(x):
        return orig_forward(x) + lora_module(x)
    return forward_with_lora

module.forward = make_forward(original_forward, lora)

# ===== 修改后 =====
def make_forward(orig_forward, lora_module):
    def forward_with_lora(x):
        return orig_forward(x) + lora_module(x)
    return forward_with_lora

module._original_forward = original_forward   # <-- 新增：保存原始 forward
module.forward = make_forward(original_forward, lora)
```

---

### P0-2: 修复 GRPO advantage 计算（全局归一化 → 组内归一化）

**文件**: `trainer/trainer_grpo.py`
**问题**: GRPO 的核心是"对同一个 prompt 的 N 个回答做组内比较"。当前代码用全局均值和标准差，导致不同难度 prompt 的回答被不公平比较。

**修改**:

```python
# trainer/trainer_grpo.py, main() 函数中,第 215-219 行

# ===== 修改前 =====
# 计算 advantages（全局 baseline，避免均值归零）
rewards_t = torch.tensor(all_rewards, device=device).view(B, args.num_gen)
baseline = rewards_t.mean()           # 全局均值作为 baseline
std_all = rewards_t.std(unbiased=False) + 1e-4
advantages = ((rewards_t - baseline) / std_all).view(-1)

# ===== 修改后 =====
# 计算 advantages（组内 baseline——GRPO 的正确做法）
rewards_t = torch.tensor(all_rewards, device=device).view(B, args.num_gen)
advantages = torch.zeros_like(rewards_t)
for i in range(B):
    group_mean = rewards_t[i].mean()
    group_std = rewards_t[i].std(unbiased=False) + 1e-4
    advantages[i] = (rewards_t[i] - group_mean) / group_std
advantages = advantages.view(-1)
```

---

### P0-3: 移除 `trainer_sft_hf.py` 重复的 `zero_grad`

**文件**: `trainer/trainer_sft_hf.py`
**问题**: 第 85-86 行连续调用了两次 `optimizer.zero_grad(set_to_none=True)`。

**修改**:

```python
# trainer/trainer_sft_hf.py, main() 函数中,第 82-86 行

# ===== 修改前 =====
if step % args.accumulation == 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)   # <-- 删除这一行

# ===== 修改后 =====
if step % args.accumulation == 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

---

### P0-4: 在测试套件中加入 KorinMind 自身模型的推理测试

**文件**: `tests/test_all.py`
**问题**: `test_inference` 函数测试的是 MiniMind2 而非 KorinMind。需要新增一个测试函数来验证 KorinMind 自身训练出的权重。

**修改**: 在 `test_inference` 函数之后（第 317 行附近）新增以下函数:

```python
def test_korinmind_inference():
    """测试 KorinMind 自身训练权重的推理能力"""
    header("5b. KorinMind 自身模型推理")
    
    from model.model import KorinMindConfig, KorinMindForCausalLM
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("model")
    config = KorinMindConfig()
    model = KorinMindForCausalLM(config)
    
    # 尝试加载预训练权重
    weight_path = "out/pretrain_full_512.pth"
    if not os.path.exists(weight_path):
        print("  [SKIP] pretrain_full_512.pth 不存在，跳过 KorinMind 推理测试")
        return
    
    weights = torch.load(weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    # 基本续写能力测试
    prompt = "人工智能是"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=20, do_sample=False,
            pad_token_id=0, eos_token_id=2
        )
    resp = tokenizer.decode(out[0], skip_special_tokens=True)
    test("KorinMind 续写非空", len(resp) > len(prompt))
    test("KorinMind 输出无 NaN", not torch.isnan(out).any())
    print(f"    Prompt: {prompt}")
    print(f"    Output: {resp[:100]}")
    
    # SFT 权重对话测试
    sft_path = "out/full_sft_512.pth"
    if os.path.exists(sft_path):
        sft_weights = torch.load(sft_path, map_location="cpu", weights_only=True)
        model.load_state_dict(sft_weights, strict=False)
        msgs = [{"role": "user", "content": "你好"}]
        chat_prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs2 = tokenizer(chat_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out2 = model.generate(
                **inputs2, max_new_tokens=30, do_sample=False,
                pad_token_id=0, eos_token_id=2
            )
        resp2 = tokenizer.decode(
            out2[0, inputs2["input_ids"].shape[1]:], skip_special_tokens=True
        )
        test("KorinMind SFT 对话非空", len(resp2.strip()) > 0)
        print(f"    Q: 你好")
        print(f"    A: {resp2[:80]}")
```

并在主函数中调用（第 357-363 行附近）:

```python
test_model_architecture()
test_training()
test_lora()
test_datasets()
test_inference()
test_korinmind_inference()    # <-- 新增
test_web_demo()
test_ablation()
```

---

## 🟡 P1 — 高优先级（影响体验、可维护性）

### P1-1: 为所有训练脚本添加验证集评估

**文件**: `trainer/trainer_pretrain.py`, `trainer/trainer_full_sft.py`, `trainer/trainer_lora.py`

在每个 trainer 的训练循环中，每隔 `log_interval` 步对验证集计算一次 perplexity。

**通用方案**（以 pretrain 为例）:

在 `train_epoch` 函数中添加验证逻辑:

```python
# 在 train_epoch 的日志打印代码块之后（约第 97 行）添加:

if step % args.log_interval == 0 and val_loader is not None:
    model.eval()
    total_val_loss = 0.0
    val_steps = 0
    with torch.no_grad():
        for val_input_ids, val_labels, val_mask in val_loader:
            val_input_ids = val_input_ids.to(args.device)
            val_labels = val_labels.to(args.device)
            val_mask = val_mask.to(args.device)
            with autocast_ctx:
                val_res = model(val_input_ids, labels=val_labels, attention_mask=val_mask)
            total_val_loss += (val_res.loss + val_res.aux_loss).item()
            val_steps += 1
            if val_steps >= 5:  # 只验证前 5 个 batch，加快速度
                break
    avg_val_loss = total_val_loss / val_steps
    val_ppl = math.exp(avg_val_loss)
    Logger(f"  >>> Val Loss: {avg_val_loss:.4f} | Val PPL: {val_ppl:.2f}")
    if wandb:
        wandb.log({"val_loss": avg_val_loss, "val_ppl": val_ppl})
    model.train()
```

同时在 `__main__` 中创建验证集:

```python
# 在创建 train_ds 之后添加:
# 按 9:1 分割训练/验证集
full_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
n_train = int(0.9 * len(full_ds))
n_val = len(full_ds) - n_train
train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val])

val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
```

---

### P1-2: 统一 `torch.load` 使用 `weights_only=True`

**文件**: 多处

| 文件 | 行号 | 修改 |
|------|------|------|
| `trainer/trainer_untils.py` | 153, 201 | `torch.load(...)` → `torch.load(..., weights_only=True)` |
| `model/model_lora.py` | 127 | `torch.load(...)` → `torch.load(..., weights_only=True)` |
| `trainer/trainer_grpo.py` | 281 | `torch.save(...)` 保持不变（保存无需 weights_only） |

---

### P1-3: 修复 DynamicCache 兼容——不静默丢弃 KV cache

**文件**: `model/model.py`, `KorinMindModel.forward` 方法, 第 717-718 行

```python
# ===== 修改前 =====
if hasattr(past_key_values, "layers"):
    past_key_values = None

# ===== 修改后 =====
if hasattr(past_key_values, "layers"):
    import warnings
    warnings.warn(
        "DynamicCache detected; converting to legacy tuple format. "
        "For better performance, use legacy cache format directly."
    )
    # 尝试从 DynamicCache 提取 legacy cache
    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()
    else:
        past_key_values = None
```

---

### P1-4: 为 `init_model` 添加文件不存在的友好报错

**文件**: `trainer/trainer_untils.py`, `init_model` 函数, 第 193-202 行

```python
# ===== 修改后 =====
if from_weight != "none":
    moe_suffix = (
        "_moe" if hasattr(lm_config, "use_moe") and lm_config.use_moe else ""
    )
    weight_path = (
        f"{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
    )

    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            f"权重文件不存在: {weight_path}\n"
            f"请确认 {save_dir}/ 目录下有对应的权重文件，"
            f"或使用 --from_weight none 从零开始训练。"
        )

    weights = torch.load(weight_path, map_location=device, weights_only=True)
    model.load_state_dict(weights, strict=False)
```

---

### P1-5: `generate_labels` 添加输入校验

**文件**: `dataset/dataset/lm_dataset.py`, `SFTDataset.generate_labels` 方法

在方法开头添加校验，防止 bos_id/eos_id 为空时进入死循环:

```python
def generate_labels(self, input_ids):
    labels = [-100] * len(input_ids)
    
    # 校验：确保 bos_id 和 eos_id 非空
    if not self.bos_id or not self.eos_id:
        return labels
    
    # ... 后续代码不变
```

---

### P1-6: 修正消融实验 MHA 注释

**文件**: `experiments/ablation.py`, 第 119-127 行

```python
# ===== 修改前 =====
# 实验 2：MHA (8Q/8KV) vs GQA (8Q/2KV)
print("实验 2: MHA — 8 Q heads + 8 KV heads (vs GQA 8/2)")
...
mha_model = build_model(num_key_value_heads=4)

# ===== 修改后 =====
# 实验 2：MHA (4Q/4KV) vs GQA (4Q/2KV)
# 注：tiny 模型使用 4 个 attention heads，MHA 即 4Q/4KV
print("实验 2: MHA — 4 Q heads + 4 KV heads (vs GQA 4/2)")
...
mha_model = build_model(num_key_value_heads=4)  # 与 num_attention_heads=4 一致 = MHA
```

---

## 🔵 P2 — 中优先级（数据与评估改进）

### P2-1: 补充真实预训练数据

**新建文件**: `dataset/download_pretrain_data.py`

```python
"""
从 HuggingFace 下载中文预训练语料
建议使用:
  - wikipedia-zh: 中文维基百科 (~1.5GB)
  - CNews: 中文新闻语料
  - belle/train_0.5M_CN: Belle 中文预训练数据
"""
import os
from datasets import load_dataset

def download_wiki(output_path="dataset/pretrain_wiki.jsonl"):
    """下载中文维基百科"""
    ds = load_dataset("wikipedia", "20220301.zh", split="train", streaming=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for item in ds:
            text = item["text"].strip()
            if len(text) > 100:  # 过滤太短的条目
                f.write('{"text": ' + json.dumps(text, ensure_ascii=False) + '}\n')
                count += 1
                if count >= 50000:  # 先下载 5 万条
                    break
    
    print(f"下载完成: {count} 条 -> {output_path}")

if __name__ == "__main__":
    import json
    download_wiki()
```

### P2-2: 替换 SFT 数据生成管线

**新建文件**: `dataset/download_sft_data.py`

```python
"""从 HuggingFace 下载真实 SFT 对话数据"""
import json
import os
from datasets import load_dataset

# 推荐数据集:
# - BelleGroup/train_0.5M_CN: 50万条中文指令数据
# - FreedomIntelligence/alpaca-gpt4-zh: GPT-4 生成的高质量中文指令

def download_belle(output_path="dataset/sft_real.jsonl", max_samples=5000):
    """下载 Belle 中文 SFT 数据"""
    ds = load_dataset("BelleGroup/train_0.5M_CN", split="train", streaming=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for item in ds:
            conversations = [
                {"role": "user", "content": item["instruction"] + ("\n" + item.get("input", "") if item.get("input") else "")},
                {"role": "assistant", "content": item["output"]},
            ]
            f.write(json.dumps({"conversations": conversations}, ensure_ascii=False) + "\n")
            count += 1
            if count >= max_samples:
                break
    
    print(f"下载完成: {count} 条 -> {output_path}")

if __name__ == "__main__":
    download_belle()
```

---

### P2-3: 删除冗余文件 `gen_sft.py`

```bash
# 在项目根目录执行
rm dataset/gen_sft.py
```

（旧版数据生成器，功能已被 `generate_sft_data.py` 覆盖）

---

## 🟢 P3 — 低优先级（工程优化）

### P3-1: 重命名 `trainer_untils.py` → `trainer_utils.py`

**注意**: 所有 import 该模块的文件都需要同步修改。

涉及文件:
- `trainer/trainer_pretrain.py` 第 31 行
- `trainer/trainer_full_sft.py` 第 34 行
- `trainer/trainer_lora.py` 第 30 行
- `trainer/trainer_grpo.py` 第 23 行
- `trainer/trainer_sft_hf.py` 第 13 行
- `experiments/ablation.py` 第 27 行

将这些文件中的 `from trainer.trainer_untils import` 全部改为 `from trainer.trainer_utils import`。

**或者**（如果不方便批量改 import），可以在 `trainer_untils.py` 所在位置创建一个 `trainer_utils.py` 作为重定向:

```python
# trainer/trainer_utils.py (新增)
# 重定向到 trainer_untils.py (保留旧文件名兼容)
from trainer.trainer_untils import *
```

---

### P3-2: 整理 `dataset/dataset/` 嵌套目录

当前结构:
```
dataset/
├── dataset/
│   └── lm_dataset.py    # 实际的数据集代码
├── generate_data.py
├── generate_sft_data.py
└── ...
```

建议结构:
```
dataset/
├── lm_dataset.py         # 移到外层
├── generate_data.py
├── generate_sft_data.py
└── ...
```

需要修改所有 `from dataset.dataset.lm_dataset import` → `from dataset.lm_dataset import`

涉及文件:
- `trainer/trainer_pretrain.py` 第 30 行
- `trainer/trainer_full_sft.py` 第 33 行
- `trainer/trainer_lora.py` 第 29 行
- `trainer/trainer_sft_hf.py` 第 12 行
- `experiments/ablation.py` 第 26 行
- `tests/test_all.py` 第 244、253 行

---

### P3-3: 补充 `pyproject.toml` 中的可选依赖

**文件**: `pyproject.toml`

```toml
# ===== 修改后 =====
[project]
name = "korinmind"
version = "0.1.0"
description = "从零制作的轻量级中文大语言模型"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "numpy>=2.0",
    "torch>=2.0",
    "transformers>=4.45",
]

[project.optional-dependencies]
train = [
    "datasets>=3.0",
    "swanlab>=0.3",
]
demo = [
    "streamlit>=1.28",
]
dev = [
    "pytest>=8.0",
]
all = [
    "korinmind[train,demo,dev]",
]
```

---

### P3-4: 添加 GitHub Actions CI 配置

**新建文件**: `.github/workflows/test.yml`

```yaml
name: KorinMind Tests

on:
  push:
    branches: [master]
  pull_request:
    branches: [master]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      
      - name: Install dependencies
        run: |
          pip install torch transformers datasets
      
      - name: Run tests
        run: |
          python tests/test_all.py
```

---

### P3-5: 修复 `web_demo.py` 的 argparse 实现

**文件**: `web_demo.py`, 第 99-114 行

```python
# ===== 修改后 =====
def main():
    import streamlit as st

    st.set_page_config(page_title="KorinMind Chat", page_icon="🧠")
    st.title("KorinMind Chat")
    st.caption("从零训练的 26M 参数中文大语言模型")

    # 使用 Streamlit 推荐的 query_params 方式获取参数
    model_type = st.query_params.get("model", "hf")
    weight_name = st.query_params.get("weight", "pretrain_full")
    lora_path = st.query_params.get("lora_path", None)
    
    # 命令行参数兜底
    if not st.query_params:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", default="hf", choices=MODEL_CHOICES)
        parser.add_argument("--weight", default="pretrain_full", type=str)
        parser.add_argument("--lora-path", default=None, type=str)
        try:
            # 只在存在 -- 分隔符时解析
            if "--" in sys.argv:
                args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
                model_type = args.model
                weight_name = args.weight
                lora_path = args.lora_path
        except (ValueError, IndexError):
            pass
    # ... 后续代码不变
```

---

### P3-6: 修复 `web_demo.py` 的权重路径硬编码

**文件**: `web_demo.py`, `load_model` 函数, 第 72 行

```python
# ===== 修改前 =====
weight_path = f"out/{weight_name}_512.pth"

# ===== 修改后 =====
# 尝试多个路径
import glob
candidates = [
    f"out/{weight_name}.pth",
    f"out/{weight_name}_512.pth",
] + glob.glob(f"out/{weight_name}*.pth")

weight_path = None
for path in candidates:
    if os.path.exists(path):
        weight_path = path
        break

if weight_path is None:
    available = glob.glob("out/*.pth")
    raise FileNotFoundError(
        f"找不到权重文件，尝试过: {candidates}\n"
        f"out/ 目录下可用的权重: {available}"
    )
```

---

### P3-7: GRPO 训练改为批量化生成

**文件**: `trainer/trainer_grpo.py`, `main` 函数, 第 193-213 行

```python
# ===== 修改后 =====
# 批量生成：将所有 prompt 一起编码
model.eval()
with torch.no_grad():
    # 批量 tokenize
    batch_inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=args.max_gen_len
    ).to(device)
    
    # 对每个 prompt 独立生成 N 个回答（num_return_sequences 只能在 batch=1 时使用）
    for b_idx in range(B):
        single_input = {k: v[b_idx:b_idx+1] for k, v in batch_inputs.items()}
        out = model.generate(
            **single_input, max_new_tokens=args.max_gen_len,
            do_sample=True, temperature=0.8, top_p=0.9, top_k=50,
            num_return_sequences=args.num_gen,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        input_len = single_input["input_ids"].shape[1]
        for i in range(args.num_gen):
            gen_ids = out[i]
            resp = tokenizer.decode(
                gen_ids[input_len:], skip_special_tokens=True
            )
            r = rule_reward(resp)
            all_responses.append(resp)
            all_rewards.append(r)
            all_ids.append(gen_ids)
model.train()
```

---

### P3-8: 为 MoE aux_loss 收集添加防护

**文件**: `model/model.py`, `KorinMindModel.forward`, 第 753-760 行

```python
# ===== 修改前 =====
aux_loss = sum(
    [
        layer.mlp.aux_loss
        for layer in self.layers
        if isinstance(layer.mlp, MoEFeedForward)
    ],
    hidden_states.new_zeros(1).squeeze(),
)

# ===== 修改后 =====
aux_loss = hidden_states.new_zeros(1).squeeze()
for layer in self.layers:
    if isinstance(layer.mlp, MoEFeedForward) and hasattr(layer.mlp, "aux_loss"):
        aux_loss = aux_loss + layer.mlp.aux_loss
```

---

## 📋 修改检查清单

按顺序完成，每完成一项打勾:

### 必须完成（P0）
- [ ] P0-1: merge_lora 保存 _original_forward
- [ ] P0-2: GRPO advantage 改为组内归一化
- [ ] P0-3: 移除 trainer_sft_hf.py 重复的 zero_grad
- [ ] P0-4: 测试套件中加入 KorinMind 自身模型测试

### 强烈建议（P1）
- [ ] P1-1: 训练脚本添加验证集评估
- [ ] P1-2: torch.load 统一使用 weights_only=True
- [ ] P1-3: DynamicCache 兼容处理改为降级+警告
- [ ] P1-4: init_model 添加友好报错
- [ ] P1-5: generate_labels 添加输入校验
- [ ] P1-6: 修正消融实验注释

### 建议（P2-P3）
- [ ] P2-1: 补充真实预训练数据
- [ ] P2-2: 替换 SFT 数据生成管线
- [ ] P2-3: 删除冗余 gen_sft.py
- [ ] P3-1: 重命名 trainer_untils.py → trainer_utils.py
- [ ] P3-2: 整理 dataset/dataset/ 嵌套目录
- [ ] P3-3: 补充 pyproject.toml 依赖
- [ ] P3-4: 添加 GitHub Actions CI
- [ ] P3-5: 修复 web_demo.py argparse
- [ ] P3-6: 修复 web_demo.py 权重路径硬编码
- [ ] P3-7: GRPO 训练改为批量化
- [ ] P3-8: MoE aux_loss 收集添加防护
