'''
Trains a LoRA for the LLama 3.1 8B-Instruct model based on information about Super Mario 64 Speedrunning

Run
$python train.py

Run as slurm job
$sbatch --job-name=train --mem=10G slurm-gpu.sh python train.py

Activate env
$source myenv/bin/activate
'''

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# Load the Dataset
dataset_path = "./data/sm64.jsonl"  # Path to JSONL file on SM64
data = load_dataset("json", data_files=dataset_path)


# Define Tokenizer and Model 
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Assign a padding token and Load Model
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto",
    load_in_8bit=True
)

# Format Dataset for Fine-Tuning
def format_example(example):
    prompt = f"### Instruction:\n{example['prompt']}\n### Input:\n{''}\n### Response:\n{example['completion']}"
    tokenized = tokenizer(prompt, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    return {"input_ids": tokenized["input_ids"][0], "attention_mask": tokenized["attention_mask"][0]}

tokenized_data = data["train"].map(format_example)

# Configure LoRA
lora_config = LoraConfig(
    r=8,               # Low-rank dimension
    lora_alpha=32,     # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Layers to apply LoRA
    lora_dropout=0.1,  # Dropout for regularization
    bias="none"
)
model = get_peft_model(model, lora_config)

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./models/lora-llama-finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_dir="./logs",
    save_strategy="epoch",
    fp16=True,  # Enable mixed precision training
    report_to="none"
)
data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True, max_length=512)

# Modify the data_collator to handle conversion to tensors
def custom_data_collator(data):
    input_ids = [torch.tensor(f["input_ids"]) for f in data]  # Convert lists to tensors
    attention_masks = [torch.tensor(f["attention_mask"]) for f in data]
    labels = input_ids  # For causal language modeling, labels are the same as input_ids
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels": torch.stack(labels),  # Explicitly add labels here
    }

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
    data_collator=custom_data_collator
)

# Train the Model
trainer.train()

# Save the Fine-Tuned Model
model.save_pretrained("./models/lora-llama-sm64")
tokenizer.save_pretrained("./models/lora-llama-sm64")

print("Model fine-tuned and saved!")