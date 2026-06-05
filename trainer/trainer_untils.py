# ============================================================================
# trainer_untils.py — 训练工具函数集合
# ============================================================================
# 为预训练脚本提供基础设施：分布式初始化、学习率调度、checkpoint 管理、
# 模型创建与加载、随机种子固定、断点续训支持等。
# ============================================================================

import os
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler


# ---------------------------------------------------------------------------
# is_main_process — 判断当前是否为分布式训练中的主进程
# ---------------------------------------------------------------------------
# 多 GPU 训练时只有 rank=0 的进程应该打印日志和保存 checkpoints
# 单 GPU / 非分布式模式下始终返回 True
# ---------------------------------------------------------------------------
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


# ---------------------------------------------------------------------------
# Logger — 只在主进程打印内容，避免 N 个 GPU 各打一遍
# ---------------------------------------------------------------------------
def Logger(content):
    if is_main_process():
        print(content)


# ---------------------------------------------------------------------------
# get_lr — 余弦退火学习率调度
# ---------------------------------------------------------------------------
# 公式：lr × (0.1 + 0.45 × (1 + cos(π × step / total)))
# - step=0 时：lr × (0.1 + 0.45 × 2) = lr × 1.0  → 初始学习率
# - step=end 时：lr × (0.1 + 0.45 × 0) = lr × 0.1  → 最终衰减到 10%
# 余弦曲线在中间阶段平滑过渡，相比线性衰减更有利于模型收敛
# ---------------------------------------------------------------------------
def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


# ---------------------------------------------------------------------------
# init_distributed_mode — 初始化 NCCL 分布式训练环境
# ---------------------------------------------------------------------------
# Windows 单 GPU 下 RANK 环境变量未设置，直接返回 0（非 DDP 模式）
# 多 GPU 下需通过 torchrun 启动，自动设置 RANK / LOCAL_RANK / WORLD_SIZE
# ---------------------------------------------------------------------------
def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 单 GPU / 非分布式模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


# ---------------------------------------------------------------------------
# setup_seed — 固定所有随机种子，确保实验可复现
# ---------------------------------------------------------------------------
def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# lm_checkpoint — 保存 / 加载训练检查点
# ---------------------------------------------------------------------------
# 保存模式（model is not None）：
#   - 保存纯模型权重 .pth（用于后续 fine-tune/推理）
#   - 保存完整 resume 文件（含 optimizer、epoch、step、wandb_id 等）
# 加载模式（model is None）：
#   - 优先加载 resume 文件（断点续训）
#   - 自动处理 GPU 数量变化导致的 step 转换
# ---------------------------------------------------------------------------
def lm_checkpoint(
    lm_config,
    weight="full_sft",
    model=None,
    optimizer=None,
    epoch=0,
    step=0,
    wandb=None,
    save_dir="checkpoints",
    **kwargs,
):
    os.makedirs(save_dir, exist_ok=True)

    # MoE 模型后缀区分，避免 dense 和 MoE 权重互相覆盖
    moe_path = "_moe" if hasattr(lm_config, "use_moe") and lm_config.use_moe else ""
    ckp_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth"
    resume_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth"

    if model is not None:
        # ── 保存模式 ──
        from torch.nn.parallel import DistributedDataParallel

        if isinstance(model, DistributedDataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        # 原子写入：先写 .tmp 再 rename，避免写入中断导致文件损坏
        ckp_tmp = ckp_path + ".tmp"
        torch.save({k: v.half() for k, v in state_dict.items()}, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)

        # 获取 wandb run ID 以便断点续训时恢复日志曲线
        wandb_id = None
        if wandb:
            if hasattr(wandb, "get_run"):
                run = wandb.get_run()
                wandb_id = getattr(run, "id", None) if run else None
            else:
                wandb_id = getattr(wandb, "id", None)

        resume_data = {
            "model": state_dict,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "wandb_id": wandb_id,
        }

        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, "state_dict"):
                    if isinstance(value, DistributedDataParallel):
                        resume_data[key] = value.module.state_dict()
                    else:
                        resume_data[key] = value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + ".tmp"
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)

    else:
        # ── 加载模式 ──
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location="cpu", weights_only=True)
            saved_ws = ckp_data.get("world_size", 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1

            # GPU 数量增加时等比例增加已训练的 step 数
            if saved_ws != current_ws:
                ckp_data["step"] = ckp_data["step"] * saved_ws // current_ws
                Logger(
                    f"GPU 数量变化 ({saved_ws}→{current_ws})，step 已自动调整为 {ckp_data['step']}"
                )

            return ckp_data
        return None


# ---------------------------------------------------------------------------
# init_model — 创建模型 + tokenizer + 可选加载预训练权重
# ---------------------------------------------------------------------------
def init_model(
    lm_config,
    from_weight="pretrain",
    tokenizer_path=None,
    save_dir="out",
    device="cuda",
):
    from transformers import AutoTokenizer
    from model.model import KorinMindForCausalLM

    # tokenizer 路径：默认使用项目 model/ 目录下的词表
    if tokenizer_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        tokenizer_path = os.path.join(project_root, "model")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 创建模型
    model = KorinMindForCausalLM(lm_config)

    # 加载预训练权重
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

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger(f"Model 可训练参数：{total_params / 1e6:.3f} 百万")

    return model.to(device), tokenizer


# ---------------------------------------------------------------------------
# SkipBatchSampler — 断点续训时跳过已训练的 batch
# ---------------------------------------------------------------------------
class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0

        for idx in self.sampler:
            batch.append(idx)

            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue  # 跳过这个 batch

                yield batch
                batch = []

        # 最后一个不完整 batch
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)
