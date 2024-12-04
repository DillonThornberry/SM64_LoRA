'''
Runs the LLaMa 3.1 8B-Instruct model with the SM64 LoRA

Run
$python run_lora.py

Run as slurm job
$sbatch --job-name=run_lora --mem=10G slurm-gpu.sh python run_lora.py

Activate env
$source myenv/bin/activate
'''

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import get_peft_model, LoraConfig


# Path to local model and LoRA configuration
model_path = "./models/lora-llama-sm64" 
lora_config_path = "./models/lora-llama-sm64" 

# Define base model and tokenizer
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the base model
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    load_in_8bit=True,  # Efficient memory usage
    device_map="auto"
)

# Load LoRA Configuration and Apply to the Model
lora_config = LoraConfig.from_pretrained(lora_config_path)
model = get_peft_model(model, lora_config)

# Set the model to evaluation mode (important for inference)
model.eval()

# Prepare the pipeline for text generation
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Step 5: Run the model to generate text
prompt = "Explain lakitu skip in sm64"
output = pipe(prompt, max_length=100)

print(output[0]["generated_text"])