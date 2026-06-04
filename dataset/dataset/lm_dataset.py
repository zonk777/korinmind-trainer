# ============================================================================
# lm_dataset.py — 预训练 / SFT / DPO / RLAIF 数据集
# ============================================================================

from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset

# 禁用 HuggingFace tokenizer 多进程并行，防止 DataLoader 死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================================================
# 全局预处理/后处理工具
# ============================================================================

def pre_processing_chat(conversations, add_system_ratio=0.2):
    """以一定概率随机插入 system prompt（数据增强）"""
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model.",
    ]
    if conversations and conversations[0].get("role") != "system":
        if random.random() < add_system_ratio:
            return [
                {"role": "system", "content": random.choice(SYSTEM_PROMPTS)}
            ] + conversations
    return conversations


def post_processing_chat(prompt_content, empty_think_ratio=0.05):
    """清理模板渲染后多余的空 <think> 块"""
    if (
        "<think>\n\n</think>\n\n" in prompt_content
        and random.random() > empty_think_ratio
    ):
        prompt_content = prompt_content.replace("<think>\n\n</think>\n\n", "")
    return prompt_content


# ============================================================================
# PretrainDataset — 预训练数据集（Next-Token Prediction）
# ============================================================================
# 数据格式：每行一个 JSON 对象 {"text": "一段文本内容"}
#
# 处理流程：
#   1. tokenize 文本（截断到 max_length-2，留位置给 BOS/EOS）
#   2. 拼接：[BOS] + token序列 + [EOS]
#   3. 右侧 PAD 补齐到 max_length
#   4. labels = input_ids 的副本，但 PAD 位置设为 -100
#      （CrossEntropyLoss 的 ignore_index=-100，PAD 不参与 loss 计算）
#   5. 返回 attention_mask（标记哪些位置是真实 token）
# ============================================================================
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 使用 HuggingFace datasets 惰性加载，不一次性读入内存
        self.samples = load_dataset("json", data_files=data_path, split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # Step 1：tokenize，预留 BOS + EOS 的位置
        tokens = self.tokenizer(
            str(sample["text"]),
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True,
        ).input_ids

        # Step 2：BOS + tokens + EOS
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]

        # Step 3：右侧 PAD 补齐
        input_ids = tokens + [self.tokenizer.pad_token_id] * (
            self.max_length - len(tokens)
        )
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # Step 4：labels = input_ids，但 PAD → -100
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        # Step 5：attention_mask
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return input_ids, labels, attention_mask


# ============================================================================
# SFTDataset — 有监督微调数据集
# ============================================================================
# 数据格式：每行 {"conversations": [{"role":"user"/"assistant"/"system","content":"..."}]}
#
# 关键设计：稀疏标签（Sparse Labels）
#   - 只有 assistant 回复的部分 label 非 -100，参与 loss 计算
#   - user/system 部分 label = -100，模型只学"如何回答"，不学"如何提问"
#   - 这样 loss 只反映模型"回答得好不好"，而不是"复述用户输入得好不好"
# ============================================================================
class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=jsonl_path, split="train")

        # 预先 tokenize assistant 回复的起始/结束标记，用于定位回复边界
        self.bos_id = tokenizer(
            f"{tokenizer.bos_token}assistant\n", add_special_tokens=False
        ).input_ids
        self.eos_id = tokenizer(
            f"{tokenizer.eos_token}\n", add_special_tokens=False
        ).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        """将多轮对话转换为 chat template 格式的字符串"""
        messages = conversations.copy()
        tools = (
            conversations[0]["functions"]
            if (
                conversations
                and conversations[0]["role"] == "system"
                and conversations[0].get("functions")
            )
            else None
        )
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, tools=tools
        )

    def generate_labels(self, input_ids):
        """
        生成稀疏标签：只有 assistant 回复参与 loss 计算。

        算法：
        1. 初始全 -100
        2. 扫描 input_ids，找到 bos_id（assistant 起始标记）
        3. 从起始位置向后找 eos_id（结束标记）
        4. 将 [start, end+eos_len) 区间设为真实 token id
        """
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]

        # 1. 随机插入 system prompt
        conversations = pre_processing_chat(sample["conversations"])

        # 2. 用 chat template 渲染
        prompt = self.create_chat_prompt(conversations)

        # 3. 清理空 <think> 块
        prompt = post_processing_chat(prompt)

        # 4. tokenize + PAD 补齐
        input_ids = self.tokenizer(prompt).input_ids[: self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 5. 生成稀疏标签
        labels = self.generate_labels(input_ids)

        # 6. attention_mask
        attention_mask = (
            torch.tensor(input_ids, dtype=torch.long) != self.tokenizer.pad_token_id
        ).long()

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            attention_mask,
        )
