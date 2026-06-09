# KorinMind — 轻量级大语言模型

KorinMind 是一个使用 PyTorch 实现的 26M 参数中文大语言模型。包含完整的训练管线：预训练、SFT 微调、LoRA 低秩适配，以及消融实验和 Web 聊天 Demo。

## 项目特点

- **现代架构**：RMSNorm / GQA / RoPE+YaRN / SwiGLU / MoE，与 Llama/Qwen 等主流模型对齐
- **完整训练管线**：预训练 → SFT 微调 → LoRA 高效微调，支持断点续训和混合精度
- **手写 LoRA**：不依赖 peft 库，从零实现低秩适配（权重文件仅 268KB）
- **消融实验**：控制变量验证每个组件的价值
- **Web Demo**：基于 Streamlit 的聊天界面，一键启动

## 模型架构

```
KorinMindForCausalLM
  ├── KorinMindModel
  │     ├── Embedding (vocab=6400, dim=512)
  │     ├── KorinMindBlock x8
  │     │     ├── RMSNorm → Attention (GQA: 8Q/2KV + RoPE + FlashAttn)
  │     │     └── RMSNorm → FeedForward (SwiGLU) or MoEFeedForward
  │     └── RMSNorm (final)
  └── lm_head (tied with Embedding)
```

| 参数 | 值 | 说明 |
|------|-----|------|
| 总参数量 | 25.83M | ~2600 万 |
| 隐藏维度 | 512 | 8 层 Transformer |
| 注意力头 | 8 Q / 2 KV | 分组查询注意力（GQA） |
| 词表大小 | 6400 | BPE 分词器 |
| 最大序列长度 | 32768 | 支持 YaRN 长度外推 |
| 位置编码 | RoPE (θ=1M) | 旋转位置编码 |
| 激活函数 | SiLU (SwiGLU) | 门控前馈网络 |
| 归一化 | RMSNorm | 比 LayerNorm 更快 |
| MoE | 可选 | 4 路由专家 + 1 共享专家 |

## 项目结构

```
korinmind/
├── model/
│   ├── model.py              # 核心：完整 Transformer 实现 (847行)
│   ├── model_lora.py         # LoRA 从零手写实现 (173行)
│   ├── tokenizer.json        # BPE 词表
│   └── tokenizer_config.json # Tokenizer 配置
│
├── dataset/
│   ├── dataset/
│   │   └── lm_dataset.py     # PretrainDataset + SFTDataset (204行)
│   ├── pretrain_data.jsonl   # 预训练数据
│   ├── sft_data.jsonl        # SFT 对话数据
│   ├── generate_data.py      # 预训练数据生成
│   └── gen_sft.py            # SFT 数据生成
│
├── trainer/
│   ├── trainer_pretrain.py   # 预训练脚本 (265行)
│   ├── trainer_full_sft.py   # 全参数 SFT 微调 (228行)
│   ├── trainer_lora.py       # LoRA 微调 (238行)
│   ├── trainer_sft_hf.py     # HF 模型 SFT 脚本 (109行)
│   └── trainer_untils.py     # 训练工具箱 (241行)
│
├── experiments/
│   ├── ablation.py           # 消融实验脚本 (215行)
│   └── ABLATION_REPORT.md    # 实验报告
│
├── web_demo.py               # Streamlit 聊天界面
├── main.py                   # 项目入口
└── pyproject.toml            # 项目配置
```

**总代码量**：约 2700 行 Python。

## 快速开始

### 环境要求

- Python >= 3.10
- PyTorch >= 2.0
- NVIDIA GPU (RTX 3060+ 推荐)

### 安装

```bash
git clone <your-repo-url>
cd korinmind
pip install torch transformers datasets streamlit huggingface_hub safetensors
```

### 启动 Web 聊天 Demo

```bash
streamlit run web_demo.py
```

首次运行会自动从 HuggingFace 缓存加载 MiniMind2-Small 模型。

### 训练自己的模型

```bash
# 1. 预训练
python main.py --data_path dataset/pretrain_data.jsonl --epochs 10 --batch_size 16

# 2. SFT 微调
python trainer/trainer_full_sft.py --data_path dataset/sft_data.jsonl --from_weight pretrain_full --epochs 3

# 3. LoRA 微调（高效，只训练 0.5% 参数）
python trainer/trainer_lora.py --data_path dataset/sft_data.jsonl --from_weight pretrain_full --rank 8
```

### 运行消融实验

```bash
python experiments/ablation.py
```

## 训练模式对比

| 模式 | 学习率 | 可训练参数 | 权重大小 | 适用场景 |
|------|--------|-----------|----------|----------|
| Pretrain | 5e-4 | 100% | 56MB | 从零学习语言规律 |
| Full SFT | 1e-5 | 100% | 56MB | 让模型学会对话 |
| LoRA (rank=4) | 5e-4 | 0.3% | 268KB | 资源受限下的高效微调 |
| LoRA (rank=8) | 5e-4 | 0.5% | ~500KB | 平衡效果与效率 |

## 消融实验结果

| 实验组 | 参数量 | Loss (50步) | 结论 |
|--------|--------|-------------|------|
| Baseline (GQA+SwiGLU+RoPE) | 1.61M | 8.3278 | 基准 |
| MHA (4 KV heads) | 1.67M | 8.3650 | GQA 参数更少效果相近 |
| ReLU FFN | 1.61M | 8.3479 | SwiGLU 优于 ReLU |
| No RoPE | 1.61M | 8.3280 | 短序列下影响小 |
| 8 Layers | 2.39M | 8.3693 | 深层需更多训练步数 |

## 关键设计决策

### 权重绑定 (Weight Tying)
`embed_tokens.weight` 和 `lm_head.weight` 共享同一块内存，节省 hidden_size × vocab_size ≈ 3.3M 参数。

### LoRA 初始化
- A 矩阵：高斯初始化 `N(0, 0.02)`
- B 矩阵：全零初始化
- 训练开始时 ΔW = 0，不破坏预训练权重

### 预归一化 (Pre-Norm)
归一化放在子层之前而非之后，训练更稳定，是现代 Transformer 的标准做法。

## 参考项目

- [MiniMind](https://github.com/jingyaogong/minimind) — 原始 MiniMind 项目
- [MokioMind](https://github.com/Wood-Q/MokioMind) — 教学版 MiniMind，架构参考
- 模型权重基于 [MiniMind2-Small](https://huggingface.co/jingyaogong/minimind2-small)

## License

本项目代码仅供学习和研究使用。
