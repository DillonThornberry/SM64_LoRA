
'''
Runs the LLaMa 3.1 8B-Instruct model with no LoRA

Run
$python run_llama.py

Run as slurm job
$sbatch --job-name=run_llama --mem=10G slurm-gpu.sh python run_llama.py

Activate env
$source myenv/bin/activate
'''

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import pandas as pd

# Define the model name
model_name = "meta-llama/Llama-3.1-8B-Instruct"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model and move it to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use FP16 for GPU efficiency
    device_map="auto"          # Automatically allocate model layers across GPUs
)

# Example prompt
prompt = "Explain lakitu skip in sm64"

# Tokenize the input and move it to the correct device
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate text
outputs = model.generate(
    **inputs,
    max_length=100,       # Limit the output length
    temperature=0.7,      # Control randomness in generation
    top_p=0.9,            # Nucleus sampling
    num_return_sequences=1  # Generate one response
)

# Decode and print the output
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Response:")
print(response)