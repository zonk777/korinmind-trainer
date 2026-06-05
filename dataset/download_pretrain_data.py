"""从 HuggingFace 下载中文预训练语料

用法: python dataset/download_pretrain_data.py

数据源:
  - wikipedia-zh: 中文维基百科 (~1.5GB)
  - CNews: 中文新闻语料
  - belle/train_0.5M_CN: Belle 中文预训练数据
"""

import json
import os
from datasets import load_dataset


def download_wiki(output_path="dataset/pretrain_wiki.jsonl", max_samples=50000):
    """下载中文维基百科"""
    ds = load_dataset("wikipedia", "20220301.zh", split="train", streaming=True)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for item in ds:
            text = item["text"].strip()
            if len(text) > 100:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                count += 1
                if count >= max_samples:
                    break

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"下载完成: {count} 条 -> {output_path} ({size_mb:.1f}MB)")


if __name__ == "__main__":
    download_wiki()
