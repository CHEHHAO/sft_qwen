# config.py
import os
import torch

# === 根目录 ===
PROJECT_ROOT = "/usr1/home/s124mdg41_08/dev/Capstone"

# === 数据路径 ===
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw/Financial News.json")
RAW_LABEL_PATH = os.path.join(DATA_DIR, "raw/Financial News Labels.json")
CLEANED_DATA_PATH = os.path.join(DATA_DIR, "preprocess/filtered_cleaned.jsonl")
PROMPT_DATA_PATH = os.path.join(DATA_DIR, "preprocess/prompt_data.json")
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "split/train_data.json")
VAL_DATA_PATH = os.path.join(DATA_DIR, "split/val_data.json")
TEST_DATA_PATH = os.path.join(DATA_DIR, "split/test_data.json")

# === 模型路径 ===
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B"
WEIGHT_MODEL_PATH = os.path.join(MODEL_DIR, "qwen_lora_emotion_weight")  # 合并后模型路径
LORA_MODEL_PATH = os.path.join(MODEL_DIR, "qwen_lora_emotion")        # LoRA adapter权重

# === 日志路径 ===
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
PREPROCESS_LOG_PATH = os.path.join(LOG_DIR, "preprocess.log")
TRAIN_WEIGHT_LOG_PATH = os.path.join(LOG_DIR, "train_weight.log")
TRAIN_LOG_PATH = os.path.join(LOG_DIR, "train.log")
EVAL_LOG_PATH = os.path.join(LOG_DIR, "evaluate.log")
INFERENCE_LOG_PATH = os.path.join(LOG_DIR, "inference.log")
API_INFERENCE_LOG_PATH = os.path.join(LOG_DIR, "inference.log")

#  === 测试结果路径 ===
RESULT_DIR = os.path.join(PROJECT_ROOT, "evaluation_results")
BASE_RESULT_PATH = os.path.join(RESULT_DIR, "qwen_base_metrics.json")
LORA_RESULT_PATH = os.path.join(RESULT_DIR, "qwen_lora_metrics.json")
WEIGHT_RESULT_PATH = os.path.join(RESULT_DIR, "qwen_lora_weight_metrics.json")

# === 训练参数（可选）===
MAX_SEQ_LENGTH = 512
MAX_TOKENS_GENERATE = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
