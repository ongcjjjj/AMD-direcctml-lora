# load_lora.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_lora_model(base_model_path, lora_path):
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)

    model = PeftModel.from_pretrained(base_model, lora_path)
    

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=100,
        temperature=0.7
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    model, tokenizer = load_lora_model(
        base_model_path="E:/aimodle/ai",
        lora_path="./lora_output/checkpoint-3"
    )
    
    while True:
        text = input("输入你的提示: ")
        print(generate_text(model, tokenizer, text))
        print("\n" + "="*50 + "\n")