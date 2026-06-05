"""从 HuggingFace 下载真实 SFT 对话数据

用法: python dataset/download_sft_data.py

数据源:
  - BelleGroup/train_0.5M_CN: 50万条中文指令数据
  - FreedomIntelligence/alpaca-gpt4-zh: GPT-4 生成的高质量中文指令
"""

import json
import os
from datasets import load_dataset


def download_belle(output_path="dataset/sft_real.jsonl", max_samples=5000):
    """下载 Belle 中文 SFT 数据"""
    ds = load_dataset("BelleGroup/train_0.5M_CN", split="train", streaming=True)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for item in ds:
            conversations = [
                {
                    "role": "user",
                    "content": item["instruction"]
                    + ("\n" + item.get("input", "") if item.get("input") else ""),
                },
                {"role": "assistant", "content": item["output"]},
            ]
            f.write(json.dumps({"conversations": conversations}, ensure_ascii=False) + "\n")
            count += 1
            if count >= max_samples:
                break

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"下载完成: {count} 条 -> {output_path} ({size_mb:.1f}MB)")


if __name__ == "__main__":
    download_belle()
