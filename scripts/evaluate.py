import re
import torch
import json
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import classification_report, confusion_matrix
from utils.utils import classify_emotion
from config.config import TEST_DATA_PATH, LORA_MODEL_PATH, BASE_MODEL_NAME, WEIGHT_MODEL_PATH, DEVICE

# MODEL_NAME = WEIGHT_MODEL_PATH + "_response/final"
# MODEL_NAME = LORA_MODEL_PATH + "_response/final"
# MODEL_NAME = WEIGHT_MODEL_PATH + "/final"
# MODEL_NAME = LORA_MODEL_PATH + "/final"
MODEL_NAME = BASE_MODEL_NAME

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    local_files_only=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    local_files_only=True
).to(DEVICE)
model.eval()


if __name__ == "__main__":

    label_text_to_id = {"negative": 0, "neutral": 1, "positive": 2}
    label_id_to_text = {v: k for k, v in label_text_to_id.items()}

    # ==== 加载测试集 ====
    test_data = load_dataset("json", data_files=TEST_DATA_PATH)["train"]

    # ==== 预测 + 评估 ====
    y_true = []
    y_pred = []

    print(f"\n🔧 Loading model from: {MODEL_NAME}\n")
    print("\n🚀 正在评估模型性能...\n")

    for item in tqdm(test_data):
        full_text = item["prompt"]
        # print(full_text)
        # 提取真实标签
        true_label_str = item["response"].strip().lower()
        true_label = label_text_to_id.get(true_label_str, 1)
        y_true.append(true_label)

        # 模型预测
        pred_label_str = classify_emotion(full_text, model, tokenizer, DEVICE, max_tokens=5)
        # print(pred_label_str)
        pred_label = label_text_to_id.get(pred_label_str, 1)
        y_pred.append(pred_label)

    # ==== 输出评估指标 ====
    print("\n📊 分类报告：")
    print(classification_report(y_true, y_pred, target_names=["negative", "neutral", "positive"]))

    print("📉 混淆矩阵：")
    print(confusion_matrix(y_true, y_pred))
