import os
import time
import logging
from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.utils import classify_emotion
from config.config import API_INFERENCE_LOG_PATH, WEIGHT_MODEL_PATH, LORA_MODEL_PATH, DEVICE

LOG_PATH = API_INFERENCE_LOG_PATH
MODEL_PATH = WEIGHT_MODEL_PATH + "_response/final"
device = DEVICE

# === 日志配置 ===
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# === 初始化 Flask 应用 ===
app = Flask(__name__)
logging.info("🚀 Flask app initialized.")

# === 加载模型 ===

logging.info(f"🔧 Loading model from: {MODEL_PATH}")
logging.info(f"🖥️ Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    local_files_only=True
).to(device)
model.eval()

logging.info("✅ Model and tokenizer loaded successfully.")

# === 路由定义 ===
@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        logging.warning("⚠️ Received empty input.")
        return jsonify({"error": "Missing input text"}), 400

    logging.info(f"📨 Received text: {text}")

    try:
        prediction = classify_emotion(text, model, tokenizer, device, cli=True)
        elapsed = round((time.time() - start_time) * 1000, 2)
        logging.info(f"✅ Prediction: {prediction} | ⏱️ Inference time: {elapsed} ms")

        return jsonify({
            "input": text,
            "prediction": prediction,
            "time_ms": elapsed
        })

    except Exception as e:
        logging.error(f"❌ Exception during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

# === 启动服务 ===
if __name__ == "__main__":
    logging.info("🚀 Starting Flask inference server...")
    app.run(host="0.0.0.0", port=8000)
