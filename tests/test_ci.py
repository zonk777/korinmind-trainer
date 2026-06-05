"""CI 轻量测试 — 不依赖 GPU / HF 缓存 / tokenizer 文件"""

import os, sys, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

P, F = 0, 0


def t(name, condition):
    global P, F
    if condition:
        P += 1
        print(f"  [PASS] {name}")
    else:
        F += 1
        print(f"  [FAIL] {name}")


# ===== 1. 模型架构 =====
print("=== 1. Model Architecture ===")
from model.model import KorinMindConfig, KorinMindForCausalLM

config = KorinMindConfig()
m = KorinMindForCausalLM(config)
total = sum(p.numel() for p in m.parameters()) / 1e6
t("params ~25.83M", abs(total - 25.83) < 0.1)
t("weight tying", id(m.model.embed_tokens.weight) == id(m.lm_head.weight))

x = torch.randint(0, 6400, (1, 10))
with torch.no_grad():
    o = m(x)
    o2 = m(x, labels=x)
t("logits shape (1,10,6400)", o.logits.shape == (1, 10, 6400))
t("logits no NaN", not torch.isnan(o.logits).any())
t("loss computed", o2.loss.item() != 0)

gen = m.generate(x[:, :2], max_new_tokens=5, do_sample=False, pad_token_id=0, eos_token_id=2)
t("generate works", gen.shape[1] > 2)

# ===== 2. Sub-components =====
print("\n=== 2. Components ===")
from model.model import RMSNorm, precompute_freqs, apply_rotary_pos_emb, repeat_kv
from model.model import Attention, FeedForward, KorinMindBlock, KorinMindModel

# RMSNorm
norm = RMSNorm(512)
y = norm(torch.randn(2, 10, 512))
t("RMSNorm shape", y.shape == (2, 10, 512))

# RoPE
cos, sin = precompute_freqs(64, end=100, rope_base=1e6)
t("precompute_freqs cos shape", cos.shape == (100, 64))
q = torch.randn(1, 10, 8, 64)
k = torch.randn(1, 10, 2, 64)
qr, kr = apply_rotary_pos_emb(q, k, cos[:10], sin[:10])
t("apply_rotary shape", qr.shape == q.shape and kr.shape == k.shape)

# repeat_kv
kv = torch.randn(1, 10, 2, 64)
kvr = repeat_kv(kv, 4)
t("repeat_kv 2->8", kvr.shape == (1, 10, 8, 64))

# Attention
attn = Attention(config)
cos_s, sin_s = precompute_freqs(64, end=32768, rope_base=1e6)
out_attn, _ = attn(torch.randn(1, 10, 512), (cos_s[:10], sin_s[:10]))
t("Attention shape", out_attn.shape == (1, 10, 512))

# FeedForward
ff = FeedForward(config)
y_ff = ff(torch.randn(1, 10, 512))
t("FeedForward shape", y_ff.shape == (1, 10, 512))
t("intermediate_size=1408", config.intermediate_size == 1408)

# Block
block = KorinMindBlock(0, config)
h, _ = block(torch.randn(1, 10, 512), (cos_s[:10], sin_s[:10]))
t("Block shape", h.shape == (1, 10, 512))

# Model
trunk = KorinMindModel(config)
h2, _, _ = trunk(torch.randint(0, 6400, (1, 10)))
t("KorinMindModel shape", h2.shape == (1, 10, 512))

# ===== 3. LoRA =====
print("\n=== 3. LoRA ===")
m2 = KorinMindForCausalLM(KorinMindConfig())
from model.model_lora import apply_lora, merge_lora

before = sum(p.numel() for p in m2.parameters() if p.requires_grad)
apply_lora(m2, rank=8)
after = sum(p.numel() for p in m2.parameters() if p.requires_grad)
t(f"LoRA trainable ~0.5% ({100*after/before:.1f}%)", after / before < 0.01)

# Check _original_forward saved
has_orig = any(
    hasattr(mod, "_original_forward")
    for _, mod in m2.named_modules()
    if hasattr(mod, "lora")
)
t("_original_forward saved", has_orig)

merge_lora(m2)
t("merge_lora no crash", True)

# ===== 4. Training utilities =====
print("\n=== 4. Trainer Utils ===")
from trainer.trainer_untils import get_lr, setup_seed, is_main_process

lr0 = get_lr(0, 1000, 5e-4)
lr_end = get_lr(1000, 1000, 5e-4)
t("LR step=0", abs(lr0 - 5e-4) < 1e-8)
t("LR step=end", abs(lr_end - 5e-5) < 1e-8)

setup_seed(42)
a = torch.randn(3)
setup_seed(42)
b = torch.randn(3)
t("seed reproducible", torch.allclose(a, b))
t("is_main_process", is_main_process())

# ===== 5. Imports =====
print("\n=== 5. Module Imports ===")
modules = [
    "trainer.trainer_pretrain",
    "trainer.trainer_full_sft",
    "trainer.trainer_lora",
    "trainer.trainer_grpo",
    "web_demo",
    "experiments.ablation",
]
for name in modules:
    try:
        __import__(name)
        t(f"import {name}", True)
    except Exception as e:
        t(f"import {name}", False)
        print(f"      {e}")

# ===== Summary =====
print(f"\n{'='*40}")
print(f"  {P} pass / {F} fail / {P+F} total")
if F > 0:
    print(f"  {F} TESTS FAILED")
    sys.exit(1)
else:
    print(f"  All tests passed!")
