import json
import os
import re
import logging
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm


INPUT_PATH = "/usr1/home/s124mdg41_08/dev/Capstone/data/raw/Financial News.json"
OUTPUT_PATH = "/usr1/home/s124mdg41_08/dev/Capstone/data/preprocess/filtered_cleaned.jsonl"
LABEL_PATH = "/usr1/home/s124mdg41_08/dev/Capstone/data/raw/Financial News Labels.json"
LOG_PATH = "/usr1/home/s124mdg41_08/dev/Capstone/logs/preprocess.log"
output_path = "/usr1/home/s124mdg41_08/dev/Capstone/models/qwen_lora_emotion_weight"
data_path = "/usr1/home/s124mdg41_08/dev/Capstone/data/preprocess/prompt_data.json"
log_path = "/usr1/home/s124mdg41_08/dev/Capstone/logs/train_weight.log"

RELATED_TYPES = {
    "Business", "Companies", "Politics", "Economy", "Tech",
    "Sustainability", "Legal", "Technology", "Real Estate", "Climate",
    "Markets", "Finance", "Personal Finance", "Crypto", "Banking",
}

SENTIMENT_MAP = {"negative": 0, "neutral": 1, "positive": 2}
SENTIMENT_NAME = {0: "negative", 1: "neutral", 2: "positive"}

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_article(entry):
    parts = [entry.get('Title', ''), entry.get('Description', ''), entry.get('Content', '')]
    combined = "\n\n".join([p.strip() for p in parts if p.strip()])
    return clean_text(combined)

def get_sentiment_vote(entity_list, article_id):
    sentiments = []
    for e in entity_list:
        sentiment = e.get("Sentiment", "").strip().lower()
        if sentiment not in SENTIMENT_MAP:
            logging.warning(f"未知情绪标签: '{sentiment}' (文章ID: {article_id})")
            continue
        sentiments.append(SENTIMENT_MAP[sentiment])

    if not sentiments:
        logging.info(f"无有效情绪标签，跳过该样本 (文章ID: {article_id})")
        return None
    return max(set(sentiments), key=sentiments.count)

def plot_sentiment_distribution(label_counter):
    labels = [SENTIMENT_NAME[i] for i in sorted(label_counter.keys())]
    counts = [label_counter[i] for i in sorted(label_counter.keys())]

    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("sentiment distribution")
    plt.axis('equal')
    plt.savefig("sentiment_distribution.png")
    print("情绪分布图已保存为 sentiment_distribution.png")


def main():
    with open(INPUT_PATH, "r") as f:
        raw_data = json.load(f)
    with open(LABEL_PATH, "r") as f:
        label_data = json.load(f)

    seen_ids = set()
    duplicate_counter = Counter()
    skipped_type_counter = Counter()
    count = 0
    skipped = defaultdict(int)
    label_counter = Counter()

    with open(OUTPUT_PATH, "w") as out:
        for entry in raw_data:
            type_news = entry.get("type_news", "").strip()
            if type_news not in RELATED_TYPES:
                skipped['type'] += 1
                skipped_type_counter[type_news] += 1
                continue

            article_id = entry.get("article_id")
            if not article_id:
                skipped['duplicate_or_missing_id'] += 1
                continue
            if article_id in seen_ids:
                duplicate_counter[article_id] += 1
                skipped['duplicate_or_missing_id'] += 1
                continue
            seen_ids.add(article_id)

            text = clean_article(entry)
            if not text:
                skipped['empty_text'] += 1
                continue

            entity_info = label_data.get(article_id, {}).get("entities", {}).get("details", [])
            label = get_sentiment_vote(entity_info, article_id)
            if label is None:
                skipped['no_valid_sentiment'] += 1
                continue
            label_counter[label] += 1

            new_entry = {
                "text": text,
                "label": label,
                "article_id": article_id
            }
            out.write(json.dumps(new_entry) + "\n")
            count += 1

    logging.info(f"处理完成，保存样本数量: {count}")
    for reason, num in skipped.items():
        logging.info(f"跳过样本 {reason}: {num} 条")

    print(f"完成数据清洗与情绪标签生成，共保存 {count} 条样本到 {OUTPUT_PATH}")
    print(f"日志已保存至 {LOG_PATH}")
    print(f"重复 article_id 数量：{sum(duplicate_counter.values())} 条")

    if skipped_type_counter:
        print("\n被跳过的 type_news 类型及数量：")
        logging.info("被跳过的 type_news 类型及数量：")
        for t, n in skipped_type_counter.most_common():
            print(f"{t or '[空值]'}: {n} 条")
            logging.info(f"{t or '[空值]'}: {n} 条")

    plot_sentiment_distribution(label_counter)

if __name__ == "__main__":
    # main()



    INPUT_PATH = "/usr1/home/s124mdg41_08/dev/Capstone/data/preprocess/filtered_cleaned.jsonl"
    OUTPUT_PATH = "/usr1/home/s124mdg41_08/dev/Capstone/data/preprocess/prompt_style_aligned.jsonl"

    SENTIMENT_NAME = {
        0: "negative",
        1: "neutral",
        2: "positive"
    }

    def convert_to_prompt(example):
        text = example["text"].strip()
        label_id = example["label"]
        label_text = SENTIMENT_NAME.get(label_id, "unknown")

        prompt = (
            "<|user|>\n"
            f"新闻内容：{text}\n"
            "请根据内容判断情绪（仅输出 positive / neutral / negative）：\n"
            "<|assistant|>\n"
            "这篇新闻的情绪是："
        )

        return {
            "prompt": prompt,
            "response": label_text
        }

    def process_dataset(input_path, output_path):
        with open(input_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        processed = []
        for item in tqdm(data):
            processed.append(convert_to_prompt(item))

        with open(output_path, "w", encoding="utf-8") as f:
            for item in processed:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"✅ Processed {len(processed)} examples -> {output_path}")


    process_dataset(INPUT_PATH, OUTPUT_PATH)


    # === 输入输出路径配置 ===
    INPUT_PATH = "/usr1/home/s124mdg41_08/dev/Capstone/data/preprocess/prompt_style_aligned.jsonl"
    SPLIT_DIR = "/usr1/home/s124mdg41_08/dev/Capstone/data/split"
    os.makedirs(SPLIT_DIR, exist_ok=True)

    # === 加载数据 ===
    dataset = load_dataset("json", data_files=INPUT_PATH, split="train")

    # === 第一阶段：10% 测试集 ===
    train_val_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_val = train_val_dataset["train"]
    test_data = train_val_dataset["test"]

    # 保存 test
    test_path = os.path.join(SPLIT_DIR, "test_data.json")
    test_data.to_json(test_path)
    print(f"✅ Saved test_data.json ({len(test_data)} samples)")

    # === 第二阶段：从 train_val 中再拆 12.5% 做验证集（即原始的 11.25%）===
    train_val_split = train_val.train_test_split(test_size=0.125, seed=42)
    train_data = train_val_split["train"]
    val_data = train_val_split["test"]

    # 保存 train 和 val
    train_path = os.path.join(SPLIT_DIR, "train_data.json")
    val_path = os.path.join(SPLIT_DIR, "val_data.json")

    train_data.to_json(train_path)
    val_data.to_json(val_path)

    print(f"✅ Saved train_data.json ({len(train_data)} samples)")
    print(f"✅ Saved val_data.json ({len(val_data)} samples)")


    # dataset = load_dataset("json", data_files=data_path, split="train")
    
    # train_val_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    # train_val = train_val_dataset["train"]
    # test_data = train_val_dataset["test"]
    # test_data.to_json("/usr1/home/s124mdg41_08/dev/Capstone/data/split/test_data.json")
    # train_val_split = train_val.train_test_split(test_size=0.125, seed=42)
    # train_data = train_val_split["train"]  
    # val_data = train_val_split["test"]
    # train_data.to_json("/usr1/home/s124mdg41_08/dev/Capstone/data/split/train_data.json")
    # val_data.to_json("/usr1/home/s124mdg41_08/dev/Capstone/data/split/val_data.json")