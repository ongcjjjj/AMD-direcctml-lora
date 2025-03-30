import torch
from transformers import (
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import LoraConfig, get_peft_model
import json
from torch.utils.data import Dataset

class DialogueDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}
    
    def __len__(self):
        return len(self.encodings.input_ids)

class AffinityRegressionModel(PreTrainedModel):
    def __init__(self, base_model):
        super().__init__(base_model.config)
        self.model = base_model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        **kwargs
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,  # 传递 labels 用于计算 loss
            **kwargs
        )

def preprocess_data(train_data_path, tokenizer):
    with open(train_data_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    
    texts = [item["对话"] for item in data]
    print("\n对话示例:")
    for i, text in enumerate(texts[:3]):
        print(f"{i+1}. {text}")
    
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=True 
    )
    inputs["labels"] = inputs["input_ids"]  # 设置 labels
    return inputs

def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModelForCausalLM.from_pretrained(model_path)
    model = AffinityRegressionModel(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def setup_lora(model):
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./lora_output")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_path)
    inputs = preprocess_data(args.train_data, tokenizer)
    train_dataset = DialogueDataset(inputs)
    
    model = setup_lora(model)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        save_steps=500,
        logging_steps=100,
        learning_rate=1e-4,
        fp16=False,  #必须关闭
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()

if __name__ == "__main__":
    main()