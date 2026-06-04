"""
用 HuggingFace MiniMind2-Small 直接做 SFT 微调
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, time, warnings, torch
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset.dataset.lm_dataset import SFTDataset
from trainer.trainer_untils import get_lr, Logger, setup_seed

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="dataset/sft_data.jsonl")
    parser.add_argument("--save_dir", default="out")
    parser.add_argument("--save_weight", default="sft_hf_5000")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=500)
    args = parser.parse_args()

    setup_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Logger(f"Device: {device}")

    # 加载 MiniMind2
    Logger("Loading MiniMind2-Small from HF cache...")
    model = AutoModelForCausalLM.from_pretrained(
        "jingyaogong/minimind2-small", local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "jingyaogong/minimind2-small", local_files_only=True
    )
    model = model.float().to(device)  # float16 权重转 float32，避免 NaN
    total = sum(p.numel() for p in model.parameters()) / 1e6
    Logger(f"Model: {total:.2f}M params")

    # 数据
    Logger(f"Loading data: {args.data_path}")
    ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    Logger(f"Dataset: {len(ds)} samples, {len(loader)} batches/epoch")

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    total_steps = args.epochs * len(loader)
    global_step = 0

    for epoch in range(args.epochs):
        start_time = time.time()
        epoch_loss = 0.0

        for step, (input_ids, labels, attention_mask) in enumerate(loader, start=1):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)

            # 学习率
            lr = get_lr(epoch * len(loader) + step, total_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids, labels=labels, attention_mask=attention_mask)
                loss = out.loss / args.accumulation

            loss.backward()

            if step % args.accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            epoch_loss += loss.item() * args.accumulation

            if step % args.log_interval == 0 or step == len(loader):
                avg_loss = epoch_loss / step
                elapsed = time.time() - start_time
                Logger(
                    f"Epoch {epoch+1}/{args.epochs} | Step {step}/{len(loader)} | "
                    f"loss={avg_loss:.4f} | lr={lr:.6f} | {elapsed:.0f}s"
                )

        # Epoch 结束保存
        save_path = f"{args.save_dir}/{args.save_weight}.pth"
        state = model.state_dict()
        torch.save({k: v.half().cpu() for k, v in state.items()}, save_path)
        Logger(f"Saved: {save_path}")

    Logger("SFT training complete!")


if __name__ == "__main__":
    main()
