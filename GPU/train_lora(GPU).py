from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
import torch_directml
import datasets
import torch
import os

# 设备配置
device = torch_directml.device() if torch_directml.is_available() else "cpu"

# 模型加载
model_path = "E:/aimodle/ai/model.safetensors"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    use_safetensors=True,
    torch_dtype=torch.float16
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 自动添加缺失的pad_token（针对某些特殊模型）
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token

# LoRA配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 加载单一数据文件
dataset = datasets.load_dataset(
    "json",
    data_files={"train": "C:/Users/Administrator/Desktop/date/data.json"},
    split="train"
)

# 自动分割验证集（20%作为验证）
dataset = dataset.train_test_split(test_size=0.2)

# 数据格式化函数
def format_conversation(example):
    return {"text": tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )}

# 数据处理
tokenized_data = dataset.map(
    format_conversation,
    remove_columns=["messages"],
).map(
    lambda x: tokenizer(
        x["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    ),
    batched=True
)

# 训练参数
training_args = TrainingArguments(
    output_dir="./single_file_output",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_steps=50,
    num_train_epochs=3,
    learning_rate=1e-4,
    fp16=True,
    gradient_accumulation_steps=4,
    optim="adamw_torch",
    report_to="none",
    use_directml=True
)

# 数据整理器
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 训练器配置
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    data_collator=collator
)

# 开始训练
trainer.train()

# 保存适配器
save_path = os.path.join(training_args.output_dir, "final_model")
model.save_pretrained(save_path)