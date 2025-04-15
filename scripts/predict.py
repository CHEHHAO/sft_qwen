import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
from utils.utils import classify_emotion
from config.config import DEVICE


def parse_args():
    parser = argparse.ArgumentParser(description="情绪分类推理 CLI")

    parser.add_argument("--text", type=str, required=True, help="输入文本")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径（包含 LoRA 权重）")
    parser.add_argument("--log_path", type=str, default=None, help="可选：保存预测日志的路径")

    return parser.parse_args()

def main():
    args = parse_args()

    # 加载 tokenizer 和模型
    device = DEVICE
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2", 
        local_files_only=True).to(device)

    model.eval()
    
    pred = classify_emotion(args.text, model, tokenizer, device, cli=True)

    # 输出结果
    print("="*40)
    print(f"📝 输入文本: {args.text}")
    print(f"📊 预测情绪: \033[92m{pred}\033[0m")  # 绿色高亮
    print("="*40)

    # 日志记录（可选）
    if args.log_path:
        os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
        with open(args.log_path, "a", encoding="utf-8") as f:
            log_line = f"[{datetime.now()}] TEXT: {args.text} --> PRED: {pred}\n"
            f.write(log_line)

if __name__ == "__main__":
    main()

