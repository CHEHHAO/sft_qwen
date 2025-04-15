import os
import json
import torch
import wandb
from collections import Counter
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from utils.utils import preprocess_jsonl, print_trainable_parameters
from config.config import TRAIN_LOG_PATH, TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH, BASE_MODEL_NAME, LORA_MODEL_PATH, DEVICE


# 设置环境变量以优化CUDA内存分配
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

WANDB_LOG = True

TRAIN_DATA_PATH = TRAIN_DATA_PATH
VAL_DATA_PATH = VAL_DATA_PATH
TEST_DATA_PATH = TEST_DATA_PATH
     
train_data = load_dataset("json", data_files=TRAIN_DATA_PATH)["train"]
val_data = load_dataset("json", data_files=VAL_DATA_PATH)["train"]

output_path = LORA_MODEL_PATH+'_response'
log_path = TRAIN_LOG_PATH
label_map = {0: "negative", 1: "neutral", 2: "positive"}

name_model = BASE_MODEL_NAME
device = DEVICE

tokenizer = AutoTokenizer.from_pretrained(name_model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    name_model, 
    torch_dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2", 
    trust_remote_code=True).to(device)
tokenizer.padding_side = "left"


# 定义模板
instruction_template = "<|user|>\n新闻内容："
response_template = "<|assistant|>\n这篇新闻的情绪是："

def convert_format(example):
    """
    转换单个样本格式：
    拼接 example 中的 "prompt" 与 "response" 字段生成 "text" 字段
    """
    # 获取字段值，如果缺失则返回空字符串
    prompt = example.get("prompt", "")
    response = example.get("response", "")
    return {"text": prompt + response}


# 使用 map 方法对每个数据集进行转换
train_data = train_data.map(convert_format, remove_columns=['prompt', 'response'])
val_data = val_data.map(convert_format, remove_columns=['prompt', 'response'])

def formatting_prompts_func(example):
    # for sft use
    # output_texts = []
    # output_texts.append(example['text'])
    return example["text"]

response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer, mlm=False)

# 训练LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.01,
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = SFTConfig(
    output_dir=output_path,
    overwrite_output_dir=True,
    learning_rate=1e-4,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    remove_unused_columns=True,  # 如果数据集中有多余字段建议设为 False
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
    max_seq_length=1024,
    packing=False,
    dataset_num_proc=16,
    dataset_batch_size=5000,
)

if WANDB_LOG:
    wandb.login()
    wandb.init(project="qwen-emotion", name="qwen-0.5b-lora", job_type="train")

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    peft_config=lora_config, #是否启用lora
    args=training_args,
    formatting_func=formatting_prompts_func,
    data_collator=data_collator,
)


trainer.train()
trainer.save_model(output_path + '/final')
tokenizer.save_pretrained(output_path + '/final') 
wandb.finish()