"""
KorinMind GRPO (Group Relative Policy Optimization) 对齐训练

核心思想：
  对每个 prompt 生成 N 个回答，组内排名计算 advantage，
  用 PPO-clip 损失优化策略，同时用 KL 惩罚防止偏离参考模型太远。

与 MiniMind 原版的区别：
  - 不用 rollout engine，直接用 model.generate()
  - 不用独立 reward model，用规则奖励函数
  - 参考模型 = SFT 模型（冻结）
"""

import os, sys, re, math, gc, argparse, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, torch.nn.functional as F, torch.distributed as dist
from torch import optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from transformers import AutoModelForCausalLM
from trainer.trainer_untils import (
    Logger, is_main_process, init_distributed_mode, setup_seed,
)

warnings.filterwarnings("ignore")


# ============================================================================
# 规则奖励函数
# ============================================================================

def rule_reward(response: str) -> float:
    """
    多维规则奖励，确保不同质量的回答有明显的分数差异
    奖励范围约 [-2, +2]，方差足够用于 GRPO advantage 计算
    """
    text = response.strip()
    L = len(text)
    score = 0.0

    # 1. 长度：阶梯奖励（而非二元）
    if L >= 80:
        score += 0.6
    elif L >= 40:
        score += 0.3
    elif L >= 15:
        score += 0.0
    else:
        score -= 0.8  # 太短

    if L > 400:
        score -= 0.2  # 太长，可能啰嗦

    # 2. 词汇多样性（基于字符级 n-gram）
    chars = list(text)
    if len(chars) >= 6:
        bigrams = [tuple(chars[i:i + 2]) for i in range(len(chars) - 1)]
        diversity = len(set(bigrams)) / len(bigrams)
        if diversity > 0.7:
            score += 0.4
        elif diversity > 0.4:
            score += 0.2
        elif diversity > 0.2:
            score -= 0.3
        else:
            score -= 0.6  # 高度重复

    # 3. 结构完整度
    has_period = text.endswith((".", "。", "!", "！", "?", "？"))
    has_content = any(c not in "。，！？；：、" for c in text[-5:])
    if has_period and has_content:
        score += 0.3
    elif not has_period and L > 30:
        score -= 0.2  # 没标点结尾

    # 4. 包含有意义的词汇（中文字符比例）
    chinese_chars = sum(1 for c in text if "一" <= c <= "鿿")
    if L > 0:
        chinese_ratio = chinese_chars / L
        if chinese_ratio > 0.3:
            score += 0.2
        elif chinese_ratio == 0 and L > 20:
            score -= 0.3  # 没有中文且较长

    # 5. 不要全是英文/数字/符号
    alphanum_ratio = sum(1 for c in text if c.isascii() and c.isalnum()) / max(L, 1)
    if alphanum_ratio > 0.9 and L > 20:
        score -= 0.3

    return score


# ============================================================================
# GRPO 数据集：只需要 prompt，不需要 answer
# ============================================================================

class GRPODataset(Dataset):
    def __init__(self, file_path: str, tokenizer, max_length: int = 512):
        from datasets import load_dataset

        self.tokenizer = tokenizer
        self.max_length = max_length
        raw = load_dataset("json", data_files=file_path, split="train")
        self.prompts = []
        for item in raw:
            # 支持两种格式：{"prompt": "..."} 或 {"conversations": [...]}
            if "prompt" in item:
                user_text = item["prompt"]
            elif "conversations" in item:
                user_msgs = [m for m in item["conversations"] if m["role"] == "user"]
                if not user_msgs:
                    continue
                user_text = user_msgs[0]["content"]
            else:
                continue

            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False, add_generation_prompt=True,
            )
            self.prompts.append(prompt)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


# ============================================================================
# 训练
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="dataset/sft_test.jsonl")
    parser.add_argument("--save_dir", default="out")
    parser.add_argument("--save_weight", default="grpo")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2, help="每批 prompt 数")
    parser.add_argument("--num_gen", type=int, default=4, help="每个 prompt 生成几个回答")
    parser.add_argument("--max_gen_len", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip 范围")
    parser.add_argument("--beta", type=float, default=0.01, help="KL 惩罚系数")
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--from_weight", default="sft_hf_5000", help="SFT 权重作为起点")
    parser.add_argument("--use_korinmind", action="store_true", help="使用 KorinMind 模型（而非 MiniMind2）")
    args = parser.parse_args()

    setup_seed(42)
    device = torch.device(args.device)

    # ====== 1. 加载模型和 tokenizer ======
    Logger("Loading model...")

    if args.use_korinmind:
        from model.model import KorinMindConfig, KorinMindForCausalLM
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("model")
        config = KorinMindConfig()

        model = KorinMindForCausalLM(config)
        weights = torch.load(f"out/{args.from_weight}_512.pth", map_location=device, weights_only=True)
        model.load_state_dict(weights, strict=False)
        model = model.float().to(device)
        model.train()

        ref_model = KorinMindForCausalLM(config)
        ref_weights = torch.load(f"out/{args.from_weight}_512.pth", map_location=device, weights_only=True)
        ref_model.load_state_dict(ref_weights, strict=False)
        ref_model = ref_model.float().to(device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
    else:
        tokenizer = AutoTokenizer.from_pretrained("jingyaogong/minimind2-small", local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained("jingyaogong/minimind2-small", local_files_only=True)
        model = model.float().to(device)
        model.train()

        ref_model = AutoModelForCausalLM.from_pretrained("jingyaogong/minimind2-small", local_files_only=True)
        ref_model = ref_model.float().to(device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    Logger(f"Policy model: {sum(p.numel()/1e6 for p in model.parameters()):.2f}M params")

    # ====== 2. 数据 ======
    ds = GRPODataset(args.data_path, tokenizer)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    Logger(f"Data: {len(ds)} prompts, {len(loader)} batches/epoch")

    # ====== 3. 优化器 ======
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * len(loader)

    # ====== 4. 训练 ======
    for epoch in range(args.epochs):
        for step, prompts in enumerate(loader, start=1):
            B = len(prompts)
            all_responses = []
            all_rewards = []
            all_ids = []
            all_masks = []

            # 生成 N 个回答（批量 tokenize + 逐 prompt 生成）
            model.eval()
            with torch.no_grad():
                batch_inputs = tokenizer(
                    prompts, return_tensors="pt", padding=True, truncation=True,
                    max_length=args.max_gen_len,
                ).to(device)
                for b_idx in range(B):
                    single_input = {k: v[b_idx:b_idx+1] for k, v in batch_inputs.items()}
                    input_len = single_input["input_ids"].shape[1]
                    out = model.generate(
                        **single_input, max_new_tokens=args.max_gen_len,
                        do_sample=True, temperature=0.8, top_p=0.9, top_k=50,
                        num_return_sequences=args.num_gen,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
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

            # 计算 advantages（组内归一化 — GRPO 的核心）
            rewards_t = torch.tensor(all_rewards, device=device).view(B, args.num_gen)
            advantages = torch.zeros_like(rewards_t)
            for i in range(B):
                group_mean = rewards_t[i].mean()
                group_std = rewards_t[i].std(unbiased=False) + 1e-4
                advantages[i] = (rewards_t[i] - group_mean) / group_std
            advantages = advantages.view(-1)

            # Pad sequences 对齐
            max_len = max(ids.shape[0] for ids in all_ids)
            padded_ids = torch.full((B * args.num_gen, max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
            pad_mask = torch.zeros((B * args.num_gen, max_len), device=device)
            for i, ids in enumerate(all_ids):
                padded_ids[i, :ids.shape[0]] = ids
                pad_mask[i, :ids.shape[0]] = 1

            # 旧策略 log prob
            with torch.no_grad():
                old_logits = model(padded_ids, attention_mask=pad_mask).logits
                old_log_probs = F.log_softmax(old_logits[:, :-1], dim=-1).gather(
                    2, padded_ids[:, 1:].unsqueeze(-1)
                ).squeeze(-1)

            # 新策略 log prob
            logits = model(padded_ids, attention_mask=pad_mask).logits
            log_probs = F.log_softmax(logits[:, :-1], dim=-1).gather(
                2, padded_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)

            # 参考模型 log prob
            with torch.no_grad():
                ref_logits = ref_model(padded_ids, attention_mask=pad_mask).logits
                ref_log_probs = F.log_softmax(ref_logits[:, :-1], dim=-1).gather(
                    2, padded_ids[:, 1:].unsqueeze(-1)
                ).squeeze(-1)

            # KL 散度
            kl_div = ref_log_probs - log_probs
            # approx KL: exp(d) - d - 1 (for small d)
            kl_penalty = torch.exp(kl_div) - kl_div - 1

            # PPO-clip 损失
            ratio = torch.exp(log_probs - old_log_probs.detach())
            surr1 = ratio * advantages.unsqueeze(1)
            surr2 = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon) * advantages.unsqueeze(1)
            per_token_loss = -(torch.min(surr1, surr2) - args.beta * kl_penalty)

            # 只算非 pad 部分
            response_mask = pad_mask[:, 1:]
            loss = (per_token_loss * response_mask).sum() / response_mask.sum().clamp(min=1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            # 日志
            if step % args.log_interval == 0 or step == len(loader):
                Logger(
                    f"Epoch {epoch+1}/{args.epochs} | Step {step}/{len(loader)} | "
                    f"loss={loss.item():.4f} | avg_reward={rewards_t.mean().item():.2f} | "
                    f"adv_std={advantages.std().item():.2f}"
                )

            # 保存
            if step % args.save_interval == 0 or step == len(loader):
                save_path = f"{args.save_dir}/{args.save_weight}.pth"
                torch.save({k: v.half().cpu() for k, v in model.state_dict().items()}, save_path)
                Logger(f"Saved: {save_path}")

            del padded_ids, logits, log_probs, old_log_probs, ref_log_probs
            gc.collect()

    Logger("GRPO training complete!")


if __name__ == "__main__":
    main()
