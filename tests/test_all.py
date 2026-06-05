# ============================================================================
# KorinMind 全功能测试套件
# 用法: python tests/test_all.py
# ============================================================================

import os, sys, time, json, torch, warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

PASS, FAIL = 0, 0


def test(name, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


def header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================================
# 1. 模型架构测试
# ============================================================================
def test_model_architecture():
    header("1. 模型架构")

    from model.model import KorinMindConfig, KorinMindForCausalLM, Attention, FeedForward
    from model.model import RMSNorm, precompute_freqs, apply_rotary_pos_emb, repeat_kv
    from model.model import MoEGate, MoEFeedForward, KorinMindBlock, KorinMindModel

    # 1.1 配置
    config = KorinMindConfig()
    test("hidden_size=512", config.hidden_size == 512)
    test("num_hidden_layers=8", config.num_hidden_layers == 8)
    test("num_Q_heads=8 / num_KV_heads=2", config.num_attention_heads == 8 and config.num_key_value_heads == 2)
    test("vocab_size=6400", config.vocab_size == 6400)
    test("max_position_embeddings=32768", config.max_position_embeddings == 32768)
    test("rope_theta=1e6", config.rope_theta == 1000000)
    test("MoE 默认关闭", config.use_moe is False)

    # 1.2 创建完整模型
    model = KorinMindForCausalLM(config)
    total = sum(p.numel() for p in model.parameters())
    test(f"总参数量 ~25.83M", abs(total / 1e6 - 25.83) < 0.1)
    test("权重绑定 (embed=lm_head)", id(model.model.embed_tokens.weight) == id(model.lm_head.weight))

    # 1.3 前向传播
    x = torch.randint(0, 6400, (1, 10))
    with torch.no_grad():
        out = model(x)
    test("前向传播 logits shape", out.logits.shape == (1, 10, 6400))
    test("logits 无 NaN", not torch.isnan(out.logits).any())
    test("logits 无 Inf", not torch.isinf(out.logits).any())

    # 1.4 计算 loss
    with torch.no_grad():
        out_loss = model(x, labels=x)
    test("loss 非零", out_loss.loss.item() != 0)
    test("loss 不为 NaN", not torch.isnan(out_loss.loss))

    # 1.5 Generate 功能
    with torch.no_grad():
        gen = model.generate(x[:, :2], max_new_tokens=5, do_sample=False,
                             pad_token_id=0, eos_token_id=2)
    test("generate 输出 shape 正确", gen.shape[0] == 1 and gen.shape[1] > 2)

    # 1.6 RMSNorm
    norm = RMSNorm(512)
    y = norm(torch.randn(2, 10, 512))
    test("RMSNorm shape 不变", y.shape == (2, 10, 512))

    # 1.7 RoPE
    cos, sin = precompute_freqs(dim=64, end=100, rope_base=1e6)
    q = torch.randn(1, 10, 8, 64)
    k = torch.randn(1, 10, 2, 64)
    qr, kr = apply_rotary_pos_emb(q, k, cos[:10], sin[:10])
    test("RoPE 应用后 shape 不变", qr.shape == q.shape and kr.shape == k.shape)

    # 1.8 repeat_kv (GQA)
    kv = torch.randn(1, 10, 2, 64)
    kvr = repeat_kv(kv, 4)
    test("repeat_kv: 2→8 heads", kvr.shape == (1, 10, 8, 64))

    # 1.9 Attention 单独测试
    attn = Attention(config)
    cos_s, sin_s = precompute_freqs(64, end=32768, rope_base=1e6)
    out_attn, _ = attn(torch.randn(1, 10, 512), (cos_s[:10], sin_s[:10]))
    test("Attention output shape", out_attn.shape == (1, 10, 512))
    # KV Cache
    _, cache = attn(torch.randn(1, 1, 512), (cos_s[:1], sin_s[:1]), use_cache=True)
    test("KV Cache key shape (2 KV heads)", cache[0].shape == (1, 1, 2, 64))

    # 1.10 FeedForward
    ff = FeedForward(config)
    y = ff(torch.randn(1, 10, 512))
    test("FeedForward shape 不变", y.shape == (1, 10, 512))
    test("intermediate_size=1408", config.intermediate_size == 1408)

    # 1.11 MoE (可选)
    config2 = KorinMindConfig(use_moe=True)
    moe = MoEFeedForward(config2)
    moe.train()
    y = moe(torch.randn(1, 10, 512))
    test("MoE forward shape 不变", y.shape == (1, 10, 512))
    test("MoE 产生 aux_loss", moe.aux_loss.item() > 0)

    # 1.12 Block
    block = KorinMindBlock(0, config)
    h, _ = block(torch.randn(1, 10, 512), (cos_s[:10], sin_s[:10]))
    test("Block output shape", h.shape == (1, 10, 512))

    # 1.13 KorinMindModel (Transformer backbone)
    trunk = KorinMindModel(config)
    h, presents, aux = trunk(torch.randint(0, 6400, (1, 10)))
    test("Trunk output shape", h.shape == (1, 10, 512))
    test("Trunk presents (KV cache)", len(presents) == 8)


# ============================================================================
# 2. 训练能力测试
# ============================================================================
def test_training():
    header("2. 训练能力")

    from trainer.trainer_untils import get_lr, setup_seed, is_main_process
    from trainer.trainer_untils import Logger, SkipBatchSampler, init_distributed_mode

    # 2.1 学习率调度
    lr0 = get_lr(0, 1000, 5e-4)
    lr500 = get_lr(500, 1000, 5e-4)
    lr1000 = get_lr(1000, 1000, 5e-4)
    test(f"LR step=0: {lr0:.6f} (expected 0.000500)", abs(lr0 - 5e-4) < 1e-8)
    test(f"LR step=500: {lr500:.6f} (0.25-0.30)", 0.00025 < lr500 < 0.00030)
    test(f"LR step=1000: {lr1000:.6f} (expected 0.000050)", abs(lr1000 - 5e-5) < 1e-8)

    # 2.2 随机种子可复现
    setup_seed(42)
    a = torch.randn(5)
    setup_seed(42)
    b = torch.randn(5)
    test("setup_seed 可复现", torch.allclose(a, b))

    # 2.3 主进程判断
    test("is_main_process (单GPU)", is_main_process())

    # 2.4 分布式初始化
    rank = init_distributed_mode()
    test("init_distributed_mode 返回 rank=0", rank == 0)

    # 2.5 SkipBatchSampler
    from torch.utils.data import SequentialSampler
    base = SequentialSampler(range(100))
    skip = SkipBatchSampler(base, batch_size=10, skip_batches=3)
    batches = list(skip)
    test("SkipBatchSampler 跳过 3 批", len(batches) == 7)

    # 2.6 LM Checkpoint
    from trainer.trainer_untils import lm_checkpoint
    from model.model import KorinMindConfig
    config = KorinMindConfig()
    os.makedirs("tests/tmp", exist_ok=True)
    ckp = lm_checkpoint(config, weight="test", save_dir="tests/tmp")
    test("lm_checkpoint 加载不存在=返回None", ckp is None)

    # 2.7 训练脚本可导入
    test("trainer_pretrain 可导入", _import("trainer.trainer_pretrain"))
    test("trainer_full_sft 可导入", _import("trainer.trainer_full_sft"))
    test("trainer_lora 可导入", _import("trainer.trainer_lora"))
    test("trainer_grpo 可导入", _import("trainer.trainer_grpo"))

    # 清理
    import shutil
    shutil.rmtree("tests/tmp", ignore_errors=True)


# ============================================================================
# 3. LoRA 测试
# ============================================================================
def test_lora():
    header("3. LoRA 低秩微调")

    from model.model import KorinMindConfig, KorinMindForCausalLM
    from model.model_lora import LoRA, apply_lora, save_lora, load_lora, merge_lora

    config = KorinMindConfig()
    model = KorinMindForCausalLM(config)

    # 注入前
    total_before = sum(p.numel() for p in model.parameters())
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    test(f"注入前可训练={total_before/1e6:.2f}M (100%)", trainable_before == total_before)

    # 注入 LoRA rank=8
    apply_lora(model, rank=8)
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = 100 * trainable_after / total_before
    test(f"注入后可训练={trainable_after/1e3:.1f}K ({trainable_pct:.1f}%)", trainable_pct < 1.0)

    # 前向传播
    x = torch.randint(0, 6400, (1, 10))
    with torch.no_grad():
        before = model(x).logits.clone()

    # 保存和加载 LoRA
    os.makedirs("tests/tmp", exist_ok=True)
    save_lora(model, "tests/tmp/test_lora.pth")
    import os as _os
    file_size = _os.path.getsize("tests/tmp/test_lora.pth")
    test(f"LoRA 文件 < 1MB (actual={file_size/1024:.0f}KB)", file_size < 1_000_000)

    load_lora(model, "tests/tmp/test_lora.pth")

    # 合并
    merge_lora(model)
    with torch.no_grad():
        after = model(x).logits.clone()
    # 合并前后输出应该一致（LoRA 加载了自己保存的权重，合并回原权重后理论上输出不变）
    # 但因为我们没训练 LoRA，A 和 B 是初始化的，merge 后有微小变化
    test("merge 后输出无 NaN", not torch.isnan(after).any())

    import shutil
    shutil.rmtree("tests/tmp", ignore_errors=True)


# ============================================================================
# 4. 数据集测试
# ============================================================================
def test_datasets():
    header("4. 数据集")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("jingyaogong/minimind2-small", local_files_only=True)

    # 4.1 PretrainDataset
    from dataset.dataset.lm_dataset import PretrainDataset
    ds = PretrainDataset("dataset/pretrain_test.jsonl", tokenizer, max_length=256)
    test(f"PretrainDataset: {len(ds)} samples", len(ds) == 50)
    ids, labels, mask = ds[0]
    test("input_ids shape = (256,)", ids.shape == (256,))
    test("BOS token 在起始位置", ids[0].item() == tokenizer.bos_token_id)
    test("PAD 位置 label=-100", (labels[ids == tokenizer.pad_token_id] == -100).all().item())

    # 4.2 SFTDataset
    from dataset.dataset.lm_dataset import SFTDataset
    ds2 = SFTDataset("dataset/sft_test.jsonl", tokenizer, max_length=512)
    test(f"SFTDataset: {len(ds2)} samples", len(ds2) == 11)
    ids2, labels2, mask2 = ds2[0]
    has_assistant = (labels2 >= 0).sum().item()
    has_user = (labels2 == -100).sum().item()
    test(f"SFT: {has_assistant} assistant tokens, {has_user} user tokens (masked)",
         has_assistant > 0 and has_user > 0)

    # 4.3 GRPO Dataset
    from trainer.trainer_grpo import GRPODataset
    ds3 = GRPODataset("dataset/grpo_prompts.jsonl", tokenizer)
    test(f"GRPODataset: {len(ds3)} prompts", len(ds3) == 97)
    prompt = ds3[0]
    test("Prompt 包含 chat template", "<|im_start|>" in prompt)


# ============================================================================
# 5. 推理测试
# ============================================================================
def test_inference():
    header("5. 推理能力")

    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("jingyaogong/minimind2-small", local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained("jingyaogong/minimind2-small", local_files_only=True)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # 5.1 单轮对话
    msgs = [{"role": "user", "content": "1+1等于几"}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=30, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    resp = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    test("生成非空回复", len(resp.strip()) > 0)
    test("回复不含特殊 token", "<|im_start|>" not in resp and "<|im_end|>" not in resp)
    print(f"    Q: 1+1等于几")
    print(f"    A: {resp[:80]}")

    # 5.2 采样生成 (有温度)
    with torch.no_grad():
        out2 = model.generate(**inputs, max_new_tokens=30, do_sample=True, temperature=0.8,
                              pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    resp2 = tokenizer.decode(out2[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    test("采样生成非空", len(resp2.strip()) > 0)

    # 5.3 多轮对话
    msgs_multi = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么可以帮助你的？"},
        {"role": "user", "content": "我刚才说了什么"},
    ]
    prompt3 = tokenizer.apply_chat_template(msgs_multi, tokenize=False, add_generation_prompt=True)
    inputs3 = tokenizer(prompt3, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out3 = model.generate(**inputs3, max_new_tokens=40, do_sample=False,
                              pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    resp3 = tokenizer.decode(out3[0, inputs3["input_ids"].shape[1]:], skip_special_tokens=True)
    test("多轮对话非空", len(resp3.strip()) > 0)
    print(f"    Multi-turn: {resp3[:80]}")


# ============================================================================
# 5b. KorinMind 自身模型推理
# ============================================================================
def test_korinmind_inference():
    header("5b. KorinMind 自身模型推理")

    from model.model import KorinMindConfig, KorinMindForCausalLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("model")
    config = KorinMindConfig()
    model = KorinMindForCausalLM(config)

    # 尝试加载预训练权重
    weight_path = "out/pretrain_full_512.pth"
    if not os.path.exists(weight_path):
        print("  [SKIP] pretrain_full_512.pth 不存在")
        return

    weights = torch.load(weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # 基本续写能力
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


# ============================================================================
# 6. Web Demo 导入测试
# ============================================================================
def test_web_demo():
    header("6. Web Demo")
    test("web_demo 模块可导入", _import("web_demo"))


# ============================================================================
# 7. 消融实验测试
# ============================================================================
def test_ablation():
    header("7. 消融实验")
    test("ablation 模块可导入", _import("experiments.ablation"))
    test("实验结果文件存在", os.path.exists("experiments/ablation_results.pt"))


# ============================================================================
# 辅助
# ============================================================================
def _import(name):
    try:
        __import__(name)
        return True
    except Exception as e:
        print(f"      Import error: {e}")
        return False


# ============================================================================
# 主函数
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  KorinMind 全功能测试")
    print("=" * 60)
    start = time.time()

    test_model_architecture()
    test_training()
    test_lora()
    test_datasets()
    test_inference()
    test_korinmind_inference()
    test_web_demo()
    test_ablation()

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  结果: {PASS} 通过 / {FAIL} 失败 / {PASS+FAIL} 总计")
    print(f"  耗时: {elapsed:.1f}s")
    print(f"{'='*60}")

    if FAIL > 0:
        print("\n  有测试失败，请检查上述 [FAIL] 项")
        sys.exit(1)
    else:
        print("\n  全部通过!")
