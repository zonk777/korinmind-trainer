# KorinMind 修改指南

基于 2026-06-05 深度评估报告生成。
**第二次评估更新 (2026-06-05)**：18 项建议中 15 项已完成 (83%)，3 项待处理。

---

## 🔴 P0 — 必须立即修复（影响功能正确性）

### P0-1: 修复 `merge_lora` 无法恢复原始 forward ✅

**文件**: `model/model_lora.py`
**状态**: 已完成 — `module._original_forward = original_forward` 已添加到 `apply_lora` 中 (第 80 行)

---

### P0-2: 修复 GRPO advantage 计算（全局归一化 → 组内归一化） ✅

**文件**: `trainer/trainer_grpo.py`
**状态**: 已完成 — 第 220-227 行改为 `for i in range(B)` 逐 prompt 计算组内 mean/std

---

### P0-3: 移除 `trainer_sft_hf.py` 重复的 `zero_grad` ✅

**文件**: `trainer/trainer_sft_hf.py`
**状态**: 已完成 — 重复调用已删除，第 85 行仅保留一次 `optimizer.zero_grad(set_to_none=True)`

---

### P0-4: 在测试套件中加入 KorinMind 自身模型的推理测试 ✅

**文件**: `tests/test_all.py`
**状态**: 已完成 — 新增 `test_korinmind_inference` 函数 (第 322-357 行)，并在 `__main__` 中调用 (第 403 行)

> ⚠️ 注意：当前实现只做了续写测试，SFT 对话测试部分未包含（FIX_GUIDE 建议中第 141-161 行的 SFT 测试代码未实际添加）。建议后续补上。

---

## 🟡 P1 — 高优先级（影响体验、可维护性）

### P1-1: 为所有训练脚本添加验证集评估 🟡

**状态**: **部分完成**

| 脚本 | 状态 | 说明 |
|------|:----:|------|
| `trainer/trainer_pretrain.py` | ✅ | `train_epoch` 接受 `val_loader`，主入口 9:1 分割数据，log_interval 时输出 Val Loss/PPL |
| `trainer/trainer_full_sft.py` | ❌ | `train_epoch` 未改造，主入口无 val_loader 创建 |
| `trainer/trainer_lora.py` | ❌ | `train_epoch` 未改造，主入口无 val_loader 创建 |

**待做**：参照 `trainer_pretrain.py:48,98-118,243-253,288,298` 的实现，为 full_sft 和 lora 两个 trainer 补齐等价逻辑。

**修改**（以 `trainer_full_sft.py` 为例）:

```python
# ==== trainer_full_sft.py ====

# 1. train_epoch 签名加上 val_loader 参数 (第 42 行)
def train_epoch(epoch, loader, iters, start_step=0, wandb=None, val_loader=None):

# 2. 在日志打印代码块之后 (第 81 行之后) 插入验证逻辑:
            if val_loader is not None:
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
                        if val_steps >= 5:
                            break
                avg_val = total_val_loss / max(val_steps, 1)
                import math as _math
                val_ppl = _math.exp(avg_val)
                Logger(f"  >>> Val Loss: {avg_val:.4f} | Val PPL: {val_ppl:.2f}")
                if wandb:
                    wandb.log({"val_loss": avg_val, "val_ppl": val_ppl})
                model.train()

# 3. 主入口创建验证集 (第 188 行 train_ds 之后):
    n_train = int(0.9 * len(train_ds))
    n_val = len(train_ds) - n_train
    val_loader = None
    if n_val > 0:
        train_ds, val_ds = torch.utils.data.random_split(train_ds, [n_train, n_val])
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )
        Logger(f"数据划分: train={n_train}, val={n_val}")

# 4. train_epoch 调用处传入 val_loader (第 219、226 行)
    train_epoch(epoch, loader, ..., start_step, wandb, val_loader)
```

`trainer_lora.py` 做相同的三处修改。

---

### P1-2: 统一 `torch.load` 使用 `weights_only=True` ✅

**状态**: 已完成

| 文件 | 状态 |
|------|:----:|
| `trainer/trainer_untils.py` 第 153, 208 行 | ✅ |
| `model/model_lora.py` 第 127 行 | ✅ |
| `web_demo.py` 第 89 行 | ✅ |
| `tests/test_all.py` 第 338 行 | ✅ |

---

### P1-3: 修复 DynamicCache 兼容 ✅

**文件**: `model/model.py`, 第 717-721 行
**状态**: 已完成 — 增加了 `to_legacy_cache()` 降级路径（未添加 warning 但功能已正确）

---

### P1-4: 为 `init_model` 添加文件不存在的友好报错 ✅

**文件**: `trainer/trainer_untils.py`, 第 201-206 行
**状态**: 已完成 — 抛出具中文提示的 `FileNotFoundError`

---

### P1-5: `generate_labels` 添加输入校验 ✅

**文件**: `dataset/dataset/lm_dataset.py`, 第 162-163 行
**状态**: 已完成 — 添加了 `if not self.bos_id or not self.eos_id: return labels` 防护

---

### P1-6: 修正消融实验 MHA 注释 ✅

**文件**: `experiments/ablation.py`, 第 119-127 行
**状态**: 已完成 — 注释已改为 `MHA — 4 Q heads + 4 KV heads (vs GQA 4/2)`

---

## 🔵 P2 — 中优先级（数据与评估改进）

### P2-1: 补充真实预训练数据 ✅

**新建文件**: `dataset/download_pretrain_data.py`
**状态**: 已完成 — 从 HuggingFace 下载中文 Wikipedia-zh 的脚本，`streaming=True` 惰性加载

---

### P2-2: 替换 SFT 数据生成管线 ✅

**新建文件**: `dataset/download_sft_data.py`
**状态**: 已完成 — 从 HuggingFace 下载 Belle 50 万条中文指令数据

---

### P2-3: 删除冗余文件 `gen_sft.py` ✅

**状态**: 已完成 — `dataset/gen_sft.py` 已删除

---

## 🟢 P3 — 低优先级（工程优化）

### P3-1: 重命名 `trainer_untils.py` → `trainer_utils.py` 🟡

**状态**: **半完成，需要进一步处理**

当前情况：
- `trainer/trainer_utils.py` 已创建（2 行重定向文件：`from trainer.trainer_untils import *`）
- **但是**所有 trainer 文件的 import 语句**仍引用 `trainer_untils`**，`trainer_utils.py` 是死文件

需要二选一：

**方案 A**（推荐）：批量修改 6 个文件的 import 语句：
- `trainer/trainer_pretrain.py` 第 31 行
- `trainer/trainer_full_sft.py` 第 34 行
- `trainer/trainer_lora.py` 第 30 行
- `trainer/trainer_grpo.py` 第 23 行
- `trainer/trainer_sft_hf.py` 第 13 行
- `experiments/ablation.py` 第 27 行

将 `from trainer.trainer_untils import` → `from trainer.trainer_utils import`，然后删除 `trainer_utils.py` 中的 `import *`，改为显式列出导出符号。

**方案 B**：删除 `trainer_utils.py`，接受 `trainer_untils.py` 的拼写，并在文件头部添加注释说明。

---

### P3-2: 整理 `dataset/dataset/` 嵌套目录 ❌

**状态**: 未完成 — `lm_dataset.py` 仍在 `dataset/dataset/lm_dataset.py`

所有 import 仍为 `from dataset.dataset.lm_dataset import`。需将文件移动到 `dataset/lm_dataset.py`，然后修改 6 个引用文件的 import 语句。

---

### P3-3: 补充 `pyproject.toml` 中的可选依赖 ✅

**状态**: 已完成 — 已添加 `[project.optional-dependencies]` 分组 (train/demo/dev/all)

---

### P3-4: 添加 GitHub Actions CI 配置 ✅

**新建文件**: `.github/workflows/test.yml`
**状态**: 已完成 — push/PR 到 master 时触发，运行 `test_all.py`

---

### P3-5: 修复 `web_demo.py` 的 argparse 实现 ✅

**文件**: `web_demo.py`, 第 122-139 行
**状态**: 已完成 — 使用 `st.query_params` 优先 + 命令行 `--` 兜底

---

### P3-6: 修复 `web_demo.py` 的权重路径硬编码 ✅

**文件**: `web_demo.py`, 第 73-88 行
**状态**: 已完成 — 支持多路径查找（glob 匹配），找不到时列出可用文件

---

### P3-7: GRPO 训练改为批量化生成 ✅

**文件**: `trainer/trainer_grpo.py`, 第 192-217 行
**状态**: 已完成 — 批量 tokenize + 逐 prompt 生成

---

### P3-8: 为 MoE aux_loss 收集添加防护 ✅

**文件**: `model/model.py`, 第 756-759 行
**状态**: 已完成 — 从 `sum([...])` 改为安全的 for loop + `hasattr` 检查

---

## 📋 修改检查清单

### 必须完成（P0）
- [x] P0-1: merge_lora 保存 _original_forward
- [x] P0-2: GRPO advantage 改为组内归一化
- [x] P0-3: 移除 trainer_sft_hf.py 重复的 zero_grad
- [x] P0-4: 测试套件中加入 KorinMind 自身模型测试

### 强烈建议（P1）
- [x] P1-1-a: trainer_pretrain.py 添加验证集评估
- [ ] P1-1-b: trainer_full_sft.py 添加验证集评估（待完成）
- [ ] P1-1-c: trainer_lora.py 添加验证集评估（待完成）
- [x] P1-2: torch.load 统一使用 weights_only=True
- [x] P1-3: DynamicCache 兼容处理改为降级
- [x] P1-4: init_model 添加友好报错
- [x] P1-5: generate_labels 添加输入校验
- [x] P1-6: 修正消融实验注释

### 建议（P2-P3）
- [x] P2-1: 补充真实预训练数据 (download_pretrain_data.py)
- [x] P2-2: 替换 SFT 数据生成管线 (download_sft_data.py)
- [x] P2-3: 删除冗余 gen_sft.py
- [ ] P3-1: trainer_utils.py — 需将 import 语句从 trainer_untils 改为 trainer_utils
- [ ] P3-2: 整理 dataset/dataset/ 嵌套目录
- [x] P3-3: 补充 pyproject.toml 依赖
- [x] P3-4: 添加 GitHub Actions CI
- [x] P3-5: 修复 web_demo.py argparse
- [x] P3-6: 修复 web_demo.py 权重路径硬编码
- [x] P3-7: GRPO 训练改为批量化
- [x] P3-8: MoE aux_loss 收集添加防护

---

## 🆕 第二次评估新增发现

以下问题在首次评估中未被识别，在第二次评估中发现：

### N1: `trainer_utils.py` 是死文件（关联 P3-1）

`trainer/trainer_utils.py` 已创建但所有文件仍 import `trainer_untils`。该重定向文件目前没有任何代码引用。处理方案见 P3-1。

### N2: `test_korinmind_inference` SFT 对话测试缺失（关联 P0-4）

当前 `test_all.py:322-357` 只测试了预训练模型的续写能力，FIX_GUIDE 中建议的 SFT 对话测试（第 141-161 行）未被包含。补充代码：

```python
# 在 test_korinmind_inference 函数末尾 (第 357 行之前) 添加:
    # SFT 权重对话测试
    sft_path = "out/full_sft_512.pth"
    if os.path.exists(sft_path):
        model = KorinMindForCausalLM(config)
        sft_weights = torch.load(sft_path, map_location="cpu", weights_only=True)
        model.load_state_dict(sft_weights, strict=False)
        model = model.to(device)
        model.eval()
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
    else:
        print("  [SKIP] full_sft_512.pth 不存在")
```

### N3: GRPO 仍使用 MiniMind2 而非 KorinMind

`trainer_grpo.py:162-163` 加载的是 `jingyaogong/minimind2-small`。GRPO 脚本本质上从未在 KorinMind 自身模型上运行过。如果需要 GRPO 对齐 KorinMind 模型，需新增 `--use_korinmind` 参数：

```python
# trainer_grpo.py, main() 中加载模型部分
parser.add_argument("--use_korinmind", action="store_true", help="使用 KorinMind 而非 MiniMind2")

# 然后在加载模型时分叉:
if args.use_korinmind:
    from model.model import KorinMindConfig, KorinMindForCausalLM
    config = KorinMindConfig()
    model = KorinMindForCausalLM(config)
    weights = torch.load(f"out/{args.from_weight}_512.pth", map_location=device, weights_only=True)
    model.load_state_dict(weights, strict=False)
    model = model.float().to(device)
    ref_model = KorinMindForCausalLM(config)
    ref_weights = torch.load(f"out/{args.from_weight}_512.pth", map_location=device, weights_only=True)
    ref_model.load_state_dict(ref_weights, strict=False)
    ref_model = ref_model.float().to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
else:
    # 现有的 MiniMind2 加载逻辑
    ...
```

---

## 📊 进度总览

| 优先级 | 总计 | ✅ 完成 | 🟡 部分 | ❌ 未完成 |
|--------|:----:|:------:|:------:|:--------:|
| P0 必须修复 | 4 | 4 | 0 | 0 |
| P1 强烈建议 | 6 | 4 | 2¹ | 0 |
| P2 建议 | 3 | 3 | 0 | 0 |
| P3 优化 | 5 | 3 | 1² | 1³ |
| 新增问题 | 3 | 0 | 0 | 3 |
| **合计** | **21** | **14** | **3** | **4** |

¹ P1-1 full_sft 和 lora 的验证集评估未完成（pretrain 已完成）
² P3-1 重定向文件已创建但 import 语句未更新
³ P3-2 目录整理未执行
