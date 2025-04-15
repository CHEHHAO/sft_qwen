import os
import json
import torch
import wandb
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from collections import Counter
from torch.nn import CrossEntropyLoss
# from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from utils.utils import print_trainable_parameters, formatting_prompts_func, convert_format
import torch.nn.functional as F
from config.config import TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH, BASE_MODEL_NAME, TRAIN_WEIGHT_LOG_PATH, WEIGHT_MODEL_PATH, DEVICE


# 设置环境变量以优化CUDA内存分配
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

label_map = {0: "negative", 1: "neutral", 2: "positive"}
inv_label_map = {v: k for k, v in label_map.items()}

WANDB_LOG = True
name_model = BASE_MODEL_NAME
device = DEVICE
output_path = WEIGHT_MODEL_PATH
log_path = TRAIN_WEIGHT_LOG_PATH
TRAIN_DATA_PATH = TRAIN_DATA_PATH
VAL_DATA_PATH = VAL_DATA_PATH
TEST_DATA_PATH = TEST_DATA_PATH

train_data = load_dataset("json", data_files=TRAIN_DATA_PATH)["train"]
val_data = load_dataset("json", data_files=VAL_DATA_PATH)["train"]

train_data = train_data.map(convert_format, remove_columns=['prompt', 'response'])
val_data = val_data.map(convert_format, remove_columns=['prompt', 'response'])

def add_label(example):
    try:
        # 从文本中抽取情感信息
        sentiment_text = example["text"].split("这篇新闻的情绪是：")[-1].strip().split("\n")[0].split("。")[0].lower()
        label_id = inv_label_map.get(sentiment_text, 1)
    except:
        label_id = 1
    example["label"] = label_id
    return example

# 为 train 分集添加 label 字段（如果 test 中也需要，可类似操作）
train_data = train_data.map(add_label)

# 先将各个类别分组
label_datasets = {}
for label in label_map.keys():
    label_datasets[label] = train_data.filter(lambda x: x["label"] == label)

# 查看各类别样本数量
for label, ds in label_datasets.items():
    print(f"标签 {label_map[label]} 的样本数：", len(ds))

# 找到最小样本数
min_count = min([len(ds) for ds in label_datasets.values()])
print("下采样的最小样本数：", min_count)

# 对每个类别随机采样 min_count 个样本，并合并
balanced_datasets = [label_datasets[label].shuffle(seed=42).select(range(min_count)) for label in label_map.keys()]
balanced_train_dataset = concatenate_datasets(balanced_datasets).shuffle(seed=42)
print("平衡后 train 集样本数：", len(balanced_train_dataset))

tokenizer = AutoTokenizer.from_pretrained(name_model, trust_remote_code=True)
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    name_model, 
    torch_dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2", 
    trust_remote_code=True)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM,
    bias="none"
)
model = get_peft_model(model, peft_config)
print("模型参数量：", print_trainable_parameters(model))

def tokenize_fn(example):
    encoded = tokenizer(
        example["text"],
        truncation=True,
        max_length=1024,
        padding="max_length",
    )
    encoded["labels"] = encoded["input_ids"].copy()
    return encoded

tokenized_train = balanced_train_dataset.map(tokenize_fn, batched=False, num_proc=16, remove_columns=["text"])
tokenized_val = val_data.map(tokenize_fn, batched=False, num_proc=16, remove_columns=["text"])


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=output_path,
    remove_unused_columns=False,
    learning_rate=1e-4,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    per_device_train_batch_size=6,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,  # 有效 batch size = 4
    num_train_epochs=5,
    eval_strategy="epoch",
    logging_dir=log_path,
    logging_steps=50,
    save_strategy="epoch",
    report_to=["wandb"],
    bf16=True,
    load_best_model_at_end=True,
    label_names=["labels"],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model(output_path+'/final')
tokenizer.save_pretrained(output_path+'/final') 
wandb.finish()
