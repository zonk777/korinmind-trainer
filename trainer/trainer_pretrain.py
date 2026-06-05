# ============================================================================
# trainer_pretrain.py — KorinMind 预训练脚本
# ============================================================================
# 从零训练一个 26M 参数的 Transformer 语言模型。
#
# 启动命令（最小测试）：
#   python trainer/trainer_pretrain.py --data_path dataset/pretrain_test.jsonl --epochs 1 --max_seq_len 128 --batch_size 4 --accumulation_steps 2 --log_interval 5 --save_interval 20
#
# 正式训练：
#   python trainer/trainer_pretrain.py --data_path dataset/pretrain_hq.jsonl --epochs 2 --batch_size 32
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
from dataset.dataset.lm_dataset import PretrainDataset
from trainer.trainer_untils import (
    get_lr,
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    init_model,
    SkipBatchSampler,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# train_epoch — 训练一个 epoch
# ---------------------------------------------------------------------------
def train_epoch(epoch, loader, iters, start_step=0, wandb=None, val_loader=None):
    start_time = time.time()

    for step, (input_ids, labels, attention_mask) in enumerate(
        loader, start=start_step + 1
    ):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        attention_mask = attention_mask.to(args.device)

        # 余弦退火学习率
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with autocast_ctx:
            # 前向传播：模型内部计算 cross-entropy loss
            res = model(input_ids, labels=labels, attention_mask=attention_mask)
            loss = (res.loss + res.aux_loss) / args.accumulation_steps

        # 反向传播（梯度累积）
        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            # 梯度裁剪（防止 loss 爆炸）
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 优化器更新 + scaler 更新
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # 打印日志
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]["lr"]
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60

            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}) "
                f"loss:{current_loss:.6f} lr:{current_lr:.10f} eta:{eta_min}min"
            )

            if wandb:
                wandb.log(
                    {"loss": current_loss, "lr": current_lr, "epoch_time": eta_min}
                )

            # 验证集评估
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
                model.train()

        # 保存 checkpoint
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()

            moe_suffix = (
                "_moe" if hasattr(lm_config, "use_moe") and lm_config.use_moe else ""
            )
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"

            if isinstance(model, DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            state_dict = {k: v.half() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)

            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir="checkpoints",
            )

            model.train()


# ============================================================================
# 主入口
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KorinMind Pretraining")

    # 基础训练参数
    parser.add_argument("--save_dir", type=str, default="out", help="模型保存目录")
    parser.add_argument("--save_weight", default="pretrain", type=str, help="权重前缀")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")

    # 硬件和性能
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")

    # 训练策略
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="保存间隔")

    # 模型架构
    parser.add_argument("--hidden_size", default=512, type=int, help="隐藏维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="层数")
    parser.add_argument("--max_seq_len", default=512, type=int, help="最大序列长度")
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1], help="是否使用 MoE")

    # 数据和恢复
    parser.add_argument("--data_path", type=str, default="dataset/pretrain_test.jsonl", help="数据路径")
    parser.add_argument("--from_weight", default="none", type=str, help="基于哪个权重训练")
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1], help="是否断点续训")

    # 实验跟踪
    parser.add_argument("--use_wandb", action="store_true", help="是否使用 wandb")
    parser.add_argument("--wandb_project", type=str, default="KorinMind-Pretrain", help="wandb 项目名")

    args = parser.parse_args()

    # ====== 1. 初始化环境 ======
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"

    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ====== 2. 模型配置 ======
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

    # ====== 3. 混合精度设置 ======
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    autocast_ctx = (
        nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    )

    # ====== 4. WandB / SwanLab ======
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None

        wandb_run_name = (
            f"KorinMind-Pretrain-E{args.epochs}-B{args.batch_size}-LR{args.learning_rate}"
        )
        wandb.init(
            project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume
        )

    # ====== 5. 模型、数据、优化器 ======
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)

    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)

    # 9:1 分割训练/验证集
    n_train = int(0.9 * len(train_ds))
    n_val = len(train_ds) - n_train
    val_ds = None
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

        # 断点续训：跳过已训练的 batch
        if epoch == start_epoch and start_step > 0:
            batch_sampler = SkipBatchSampler(
                train_sampler or range(len(train_ds)), args.batch_size, start_step
            )
            loader = DataLoader(
                train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True
            )
            Logger(f"Epoch [{epoch+1}/{args.epochs}]: 跳过前 {start_step} 个 step，从 step {start_step+1} 开始")
            train_epoch(epoch, loader, len(loader) + start_step, start_step, wandb, val_loader)
        else:
            loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            train_epoch(epoch, loader, len(loader), 0, wandb, val_loader)

    Logger("Training finished!")
