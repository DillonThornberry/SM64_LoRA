# SM64_LoRA
A program for training a LoRA for the [LLaMa 3.1 8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) model to have domain-specific knowledge. The knowledge used in this was information about how to speedrun Super Mario 64 using data scraped from ukikipedia.net, the main source for SM64 speedrunning info

## To run: 
(Model must be obtained from [HuggingFace](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) in order to run)

(Info for running on a cluster with Slurm included in file comments)
### Training: 
`$ python train.py`
### Inference:
#### With LoRA:
`$ python run-lora.py`
#### Without LoRA:
`$ python run-llama.py`

## Data retrieval and preprocessing pipeline (from /preprocessing)
1. `getlinks.py` - Scrapes relevant links from wiki -> `links.txt`
2. `getwikitext.py` - Scrapes relevant text from links -> `wiki_text.json`
3. `cleanjson.py` - Removes irrelevant text and redundant chars from data ->  `wiki_clean.json`
4. `finalizedata.py` - Replaces special chars and puts text in final format for training -> `data/sm64.jsonl`




