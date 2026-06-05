# ============================================================================
# trainer_lora.py — KorinMind LoRA 低秩微调
# ============================================================================
# LoRA（Low-Rank Adaptation）：只训练新增的小矩阵，冻结原始模型。
# 参数量仅增加 ~2%，训练速度比 full SFT 快 3-5 倍。
#
# 启动命令：
#   python trainer/trainer_lora.py --data_path dataset/sft_test.jsonl --epochs 3 --batch_size 8 --rank 8
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
from model.model_lora import apply_lora, save_lora
from dataset.dataset.lm_dataset import SFTDataset
from trainer.trainer_untils import (
    get_lr, Logger, is_main_process, lm_checkpoint,
    init_distributed_mode, setup_seed, init_model, SkipBatchSampler,
)

warnings.filterwarnings("ignore")


def train_epoch(epoch, loader, iters, lora_params, start_step=0, wandb=None, val_loader=None):
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
            res = model(input_ids, labels=labels, attention_mask=attention_mask)
            loss = (res.loss + res.aux_loss) / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
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
                f"loss:{current_loss:.4f} lr:{current_lr:.6f} eta:{eta_min}min"
            )

            if wandb:
                wandb.log({"loss": current_loss, "lr": current_lr})

            if val_loader is not None:
                model.eval()
                total_val_loss = 0.0
                val_steps = 0
                with torch.no_grad():
                    for val_ids, val_labs, val_mask in val_loader:
                        val_ids = val_ids.to(args.device)
                        val_labs = val_labs.to(args.device)
                        val_mask = val_mask.to(args.device)
                        with autocast_ctx:
                            val_res = model(val_ids, labels=val_labs, attention_mask=val_mask)
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

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()

            raw_model = (
                model.module if isinstance(model, DistributedDataParallel) else model
            )
            raw_model = getattr(raw_model, "_orig_mod", raw_model)

            # 只保存 LoRA 权重
            lora_path = (
                f"{args.save_dir}/{args.save_weight}_rank{args.rank}"
                f"_{lm_config.hidden_size}.pth"
            )
            save_lora(raw_model, lora_path)

            lm_checkpoint(
                lm_config, weight=args.save_weight, model=model,
                optimizer=optimizer, scaler=scaler, epoch=epoch,
                step=step, wandb=wandb, save_dir="checkpoints",
            )

            model.train()

        del input_ids, labels, res, loss


# ============================================================================
# 主入口
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KorinMind LoRA Fine-tuning")

    parser.add_argument("--save_dir", type=str, default="out")
    parser.add_argument("--save_weight", default="lora_sft", type=str, help="权重前缀")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="LoRA 用较高 lr")

    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=50)

    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--max_seq_len", default=1024, type=int)
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1])

    # LoRA 专属参数
    parser.add_argument("--rank", default=8, type=int, help="LoRA 秩（越大效果越好，参数越多）")
    parser.add_argument("--alpha", default=16.0, type=float, help="LoRA 缩放系数")

    parser.add_argument("--data_path", type=str, default="dataset/sft_test.jsonl")
    parser.add_argument("--from_weight", default="pretrain", type=str)
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1])

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="KorinMind-LoRA")

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
            name=f"KorinMind-LoRA-R{args.rank}-E{args.epochs}",
            id=wandb_id, resume=resume,
        )

    # ====== 5. 模型 + LoRA + 数据 ======
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)

    # 注入 LoRA 适配器
    apply_lora(model, rank=args.rank, alpha=args.alpha)

    # 统计可训练参数
    lora_params = [p for p in model.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in lora_params)
    total = sum(p.numel() for p in model.parameters())
    Logger(f"LoRA rank={args.rank}: 可训练 {trainable/1e3:.1f}K / 总计 {total/1e6:.2f}M "
           f"({100*trainable/total:.1f}%)")

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)

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

    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"], strict=False)
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ====== 6. 训练 ======
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
            train_epoch(epoch, loader, len(loader) + start_step, lora_params, start_step, wandb, val_loader)
        else:
            loader = DataLoader(
                train_ds, batch_size=args.batch_size,
                shuffle=(train_sampler is None), sampler=train_sampler,
                num_workers=args.num_workers, pin_memory=True,
            )
            train_epoch(epoch, loader, len(loader), lora_params, 0, wandb, val_loader)

    Logger("LoRA Training finished!")
