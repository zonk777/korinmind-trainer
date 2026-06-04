# ============================================================================
# web_demo.py — KorinMind 聊天 Web Demo (Streamlit)
# ============================================================================
# 用法：
#   streamlit run web_demo.py                                    # 默认用 MiniMind2
#   streamlit run web_demo.py -- --model hf                      # HuggingFace 模型
#   streamlit run web_demo.py -- --model local --weight pretrain_full  # 本地权重
# ============================================================================

import argparse
import sys
import torch
from transformers import AutoTokenizer

MODEL_CHOICES = ["hf", "local"]

# ============================================================================
# 模型加载
# ============================================================================

@torch.no_grad()
def generate_response(model, tokenizer, prompt: str, max_tokens: int = 256) -> str:
    """用模型生成回复"""
    # 构造对话格式：<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    response_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response.strip() or "[模型未返回内容]"


def load_model(model_type: str, weight_name: str = None, lora_path: str = None):
    """加载模型

    model_type='hf': 从本地缓存加载 MiniMind2-Small
    model_type='local': 从本地 .pth 权重加载 KorinMind
    """
    if model_type == "hf":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            "jingyaogong/minimind2-small", local_files_only=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "jingyaogong/minimind2-small", local_files_only=True,
        )
        # MiniMind2-Small 已经是训练好的对话模型，直接使用
        total = sum(p.numel() for p in model.parameters()) / 1e6

    else:
        from model.model import KorinMindConfig, KorinMindForCausalLM

        config = KorinMindConfig()
        model = KorinMindForCausalLM(config)

        weight_path = f"out/{weight_name}_512.pth"
        weights = torch.load(weight_path, map_location="cpu", weights_only=True)
        model.load_state_dict(weights, strict=False)

        if lora_path:
            import re
            from model.model_lora import apply_lora, load_lora

            rank_match = re.search(r"rank(\d+)", lora_path)
            rank = int(rank_match.group(1)) if rank_match else 8
            apply_lora(model, rank=rank)
            load_lora(model, lora_path)

        tokenizer = AutoTokenizer.from_pretrained("model")
        total = sum(p.numel() for p in model.parameters()) / 1e6

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    return model, tokenizer, total


# ============================================================================
# Streamlit UI
# ============================================================================

def main():
    import streamlit as st

    st.set_page_config(page_title="KorinMind Chat", page_icon="🧠")
    st.title("KorinMind Chat")
    st.caption("从零训练的 26M 参数中文大语言模型")

    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="hf", choices=MODEL_CHOICES, help="hf=MiniMind2, local=KorinMind")
    parser.add_argument("--weight", default="pretrain_full", type=str, help="本地权重名")
    parser.add_argument("--lora-path", default=None, type=str)
    try:
        args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    except (ValueError, IndexError):
        args = parser.parse_args([])

    @st.cache_resource
    def get_model(model_type, weight_name, lora_path):
        return load_model(model_type, weight_name, lora_path)

    with st.spinner("正在加载模型..."):
        model, tokenizer, total_params = get_model(args.model, args.weight, args.lora_path)
    st.success(f"模型已就绪：{total_params:.2f}M 参数 | 来源：{'MiniMind2-Small' if args.model == 'hf' else 'KorinMind'}")

    # 聊天历史
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 用户输入
    if prompt := st.chat_input("输入你的问题..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                response = generate_response(model, tokenizer, prompt)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # 侧边栏
    with st.sidebar:
        st.header("关于 KorinMind")
        st.markdown("""
        - 架构：Transformer Decoder
        - 参数：26M
        - 组件：RMSNorm / GQA / RoPE / SwiGLU
        - 词表：6400 BPE
        """)
        st.divider()
        st.caption("从零手写，不依赖 HuggingFace 模型实现")


if __name__ == "__main__":
    main()
