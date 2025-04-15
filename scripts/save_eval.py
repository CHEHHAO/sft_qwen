import json
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from pathlib import Path
import matplotlib.pyplot as plt

from config.config import TEST_DATA_PATH, RESULT_DIR, DEVICE
from utils.utils import classify_emotion

RESULTS_DIR = RESULT_DIR
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

# 数据处理及评估函数
def evaluate(model_name_or_path, result_prefix):
    label_text_to_id = {"negative": 0, "neutral": 1, "positive": 2}
    label_id_to_text = {v: k for k, v in label_text_to_id.items()}

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = model.to(DEVICE)
    model.eval()

    test_data = load_dataset("json", data_files=TEST_DATA_PATH)["train"]

    y_true, y_pred, texts = [], [], []

    for item in tqdm(test_data):
        prompt = item["prompt"]
        texts.append(prompt)

        true_label_str = item["response"].strip()
        true_label = label_text_to_id[true_label_str]
        y_true.append(true_label)

        pred_label_str = classify_emotion(prompt, model, tokenizer, DEVICE, max_tokens=5)
        pred_label = label_text_to_id.get(pred_label_str, 1)
        y_pred.append(pred_label)

    df_results = pd.DataFrame({
        "text": texts,
        "true_label": [label_id_to_text[i] for i in y_true],
        "pred_label": [label_id_to_text.get(i, "unknown") for i in y_pred]
    })
    df_results.to_csv(f"{RESULTS_DIR}/{result_prefix}_evaluation_results.csv", index=False)

    report = classification_report(y_true, y_pred, target_names=["negative", "neutral", "positive"], output_dict=True)
    matrix = confusion_matrix(y_true, y_pred).tolist()
    with open(f"{RESULTS_DIR}/{result_prefix}_metrics.json", "w") as f:
        json.dump({"classification_report": report, "confusion_matrix": matrix}, f, indent=2, ensure_ascii=False)

# ==== 对比模块支持多个模型 ====
def compare_metrics(paths, labels, output_path):
    comparison = {"metric": []}
    for label in labels:
        comparison[label] = []

    metric_data = [json.load(open(p)) for p in paths]

    for label in ["negative", "neutral", "positive", "macro avg", "weighted avg"]:
        for metric in ["precision", "recall", "f1-score"]:
            comparison["metric"].append(f"{label}_{metric}")
            for m in metric_data:
                comparison_label = m["classification_report"].get(label, {}).get(metric, None)
                comparison[labels[metric_data.index(m)]].append(comparison_label)

    pd.DataFrame(comparison).to_csv(output_path, index=False)

# ==== 三模型对比绘图 ====
def plot_f1_comparison(csv_path, save_path=None):
    df = pd.read_csv(csv_path)
    f1_df = df[df["metric"].str.endswith("f1-score")].copy()
    f1_df["label"] = f1_df["metric"].str.replace("_f1-score", "")
    x = range(len(f1_df))
    bar_width = 0.15

    plt.figure(figsize=(14, 6))
    for i, column in enumerate(f1_df.columns[1:-1]):
        plt.bar([j + i * bar_width - bar_width for j in x], f1_df[column], width=bar_width, label=column)

    plt.xticks(ticks=x, labels=f1_df["label"])
    plt.ylabel("F1-score")
    plt.title("F1-score Comparison Across Models")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"📈 图表已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    from config.config import (
        BASE_MODEL_NAME, LORA_MODEL_PATH,
        WEIGHT_MODEL_PATH
    )

    evaluate(BASE_MODEL_NAME, "qwen_base")
    evaluate(LORA_MODEL_PATH + "/final", "qwen_lora")
    evaluate(f"{WEIGHT_MODEL_PATH}/final", "qwen_lora_weight")
    evaluate(LORA_MODEL_PATH + "_response/final", "qwen_lora_response")
    evaluate(WEIGHT_MODEL_PATH + "_response/final", "qwen_lora_weight_response")

    compare_metrics(
        [
            f"{RESULTS_DIR}/qwen_base_metrics.json",
            f"{RESULTS_DIR}/qwen_lora_metrics.json",
            f"{RESULTS_DIR}/qwen_lora_weight_metrics.json",
            f"{RESULTS_DIR}/qwen_lora_response_metrics.json",
            f"{RESULTS_DIR}/qwen_lora_weight_response_metrics.json",
        ],
        ["base", "lora", "lora_weight", "lora_response", "lora_weight_response"],
        f"{RESULTS_DIR}/metrics_comparison.csv"
    )

    plot_f1_comparison(
        f"{RESULTS_DIR}/metrics_comparison.csv",
        save_path=f"{RESULTS_DIR}/f1_score_comparison.png"
    )
