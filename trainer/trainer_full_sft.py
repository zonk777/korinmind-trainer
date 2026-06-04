# ============================================================================
# trainer_full_sft.py — KorinMind 全参数 SFT 微调
# ============================================================================
# 在预训练权重基础上，用对话数据微调，让模型学会"聊天"。
#
# 关键区别（vs 预训练）：
#   - 数据集：对话格式（含 user/assistant 角色），而非纯文本
#   - 标签策略：只计算 assistant 回复部分的 loss（稀疏标签）
#   - 学习率：1e-5（远低于预训练的 5e-4），防止遗忘预训练知识
#   - 序列长度：1024（对话比单段文本更长）
#
# 启动命令：
#   python trainer/trainer_full_sft.py --data_path dataset/sft_test.jsonl --epochs 3 --batch_size 4
# ============================================================================

import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from model.model import KorinMindConfig
from dataset.dataset.lm_dataset import SFTDataset
from trainer.trainer_untils import (
    get_lr, Logger, is_main_process, lm_checkpoint,
    init_distributed_mode, setup_seed, init_model, SkipBatchSampler,
)

warnings.filterwarnings("ignore")


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()

    for step, (input_ids, labels, attention_mask) in enumerate(
        loader, start=start_step + 1
    ):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        attention_mask = attention_mask.to(args.device)

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with autocast_ctx:
            # 模型内部根据 labels 计算 loss（只有 assistant 部分参与）
            res = model(input_ids, labels=labels, attention_mask=attention_mask)
            loss = (res.loss + res.aux_loss) / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]["lr"]
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60

            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}) "
                f"loss:{current_loss:.4f} lr:{current_lr:.8f} eta:{eta_min}min"
            )

            if wandb:
                wandb.log({"loss": current_loss, "lr": current_lr})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()

            moe_suffix = "_moe" if lm_config.use_moe else ""
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"

            raw_model = (
                model.module if isinstance(model, DistributedDataParallel) else model
            )
            raw_model = getattr(raw_model, "_orig_mod", raw_model)
            state_dict = raw_model.state_dict()

            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)

            lm_checkpoint(
                lm_config, weight=args.save_weight, model=model,
                optimizer=optimizer, scaler=scaler, epoch=epoch,
                step=step, wandb=wandb, save_dir="checkpoints",
            )

            model.train()
            del state_dict

        del input_ids, labels, res, loss


# ============================================================================
# 主入口
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KorinMind Full SFT")

    parser.add_argument("--save_dir", type=str, default="out", help="模型保存目录")
    parser.add_argument("--save_weight", default="full_sft", type=str, help="权重前缀")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="学习率（SFT 用低 lr）")

    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度")
    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--accumulation_steps", type=int, default=4, help="梯度累积")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--log_interval", type=int, default=5, help="日志间隔")
    parser.add_argument("--save_interval", type=int, default=50, help="保存间隔")

    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--max_seq_len", default=1024, type=int, help="对话序列更长")
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1])

    parser.add_argument("--data_path", type=str, default="dataset/sft_test.jsonl")
    parser.add_argument("--from_weight", default="pretrain", type=str, help="基于预训练权重微调")
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1])

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="KorinMind-SFT")

    args = parser.parse_args()

    # ====== 1. 初始化 ======
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"

    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ====== 2. 配置 ======
    os.makedirs(args.save_dir, exist_ok=True)

    lm_config = KorinMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )

    ckp_data = (
        lm_checkpoint(lm_config, weight=args.save_weight, save_dir="checkpoints")
        if args.from_resume == 1
        else None
    )

    # ====== 3. 混合精度 ======
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = (
        nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    )

    # ====== 4. WandB ======
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        wandb.init(
            project=args.wandb_project,
            name=f"KorinMind-SFT-E{args.epochs}-LR{args.learning_rate}",
            id=wandb_id, resume=resume,
        )

    # ====== 5. 模型、数据、优化器 ======
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ====== 6. 训练循环 ======
    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        if epoch == start_epoch and start_step > 0:
            batch_sampler = SkipBatchSampler(
                train_sampler or range(len(train_ds)), args.batch_size, start_step
            )
            loader = DataLoader(
                train_ds, batch_sampler=batch_sampler,
                num_workers=args.num_workers, pin_memory=True,
            )
            Logger(f"Epoch [{epoch+1}/{args.epochs}]: skip {start_step} steps")
            train_epoch(epoch, loader, len(loader) + start_step, start_step, wandb)
        else:
            loader = DataLoader(
                train_ds, batch_size=args.batch_size,
                shuffle=(train_sampler is None), sampler=train_sampler,
                num_workers=args.num_workers, pin_memory=True,
            )
            train_epoch(epoch, loader, len(loader), 0, wandb)

    Logger("SFT Training finished!")
