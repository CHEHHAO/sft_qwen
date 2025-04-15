import os
import json
import torch
import wandb
from collections import Counter
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from utils.utils import preprocess_jsonl, print_trainable_parameters
from config.config import TRAIN_LOG_PATH, TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH, BASE_MODEL_NAME, LORA_MODEL_PATH, DEVICE


# 设置环境变量以优化CUDA内存分配
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
label_map = {0: "negative", 1: "neutral", 2: "positive"}

WANDB_LOG = True

TRAIN_DATA_PATH = TRAIN_DATA_PATH
VAL_DATA_PATH = VAL_DATA_PATH
TEST_DATA_PATH = TEST_DATA_PATH
output_path = LORA_MODEL_PATH
log_path = TRAIN_LOG_PATH
name_model = BASE_MODEL_NAME
device = DEVICE

tokenizer = AutoTokenizer.from_pretrained(name_model, trust_remote_code=True)
tokenizer.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(
    name_model, 
    torch_dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2", 
    trust_remote_code=True).to(device)

def convert_format(example):
    """
    转换单个样本格式：
    拼接 example 中的 "prompt" 与 "response" 字段生成 "text" 字段
    """
    # 获取字段值，如果缺失则返回空字符串
    prompt = example.get("prompt", "")
    response = example.get("response", "")
    return {"text": prompt + response}

# 加载数据集
train_data = load_dataset("json", data_files=TRAIN_DATA_PATH)["train"]
test_data = load_dataset("json", data_files=TEST_DATA_PATH)["train"]
val_data = load_dataset("json", data_files=VAL_DATA_PATH)["train"]

# 使用 map 方法对每个数据集进行转换
train_data = train_data.map(convert_format, remove_columns=['prompt', 'response'])
test_data = test_data.map(convert_format, remove_columns=['prompt', 'response'])
val_data = val_data.map(convert_format, remove_columns=['prompt', 'response'])


def tokenize_fn(example):
    encoded = tokenizer(
        example["text"],
        truncation=True,
        max_length=1024,
        padding="max_length",
        return_tensors="pt"
    )
    encoded = {k: v.squeeze(0) for k, v in encoded.items()}
    encoded["labels"] = encoded["input_ids"].clone()
    return encoded

tokenized_trian = train_data.map(tokenize_fn, batched=False, num_proc=16, remove_columns=["text"])
tokenized_val = val_data.map(tokenize_fn, batched=False, num_proc=16, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors='pt')

# LoRA 配置
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
    bias="none"
)

print_trainable_parameters(model)

training_args = SFTConfig(
    output_dir=output_path,
    overwrite_output_dir=True,
    learning_rate=1e-4,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    remove_unused_columns=False,
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
    label_names=["labels"]
)

if WANDB_LOG:
    wandb.init(project="qwen-emotion", name="qwen-0.5b-lora", job_type="train")

trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_trian,
    eval_dataset=tokenized_val,
    peft_config=peft_config, #是否启用lora
    args=training_args,
    data_collator=data_collator,
)


if __name__ == "__main__":

    trainer.train()
    trainer.save_model(output_path + '/final')
    tokenizer.save_pretrained(output_path+'/final') 
    wandb.finish()
