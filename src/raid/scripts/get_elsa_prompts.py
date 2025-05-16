import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# get IDs from filenames
ids = sorted([path.stem for path in Path("data/ELSA_D3_1000/real").iterdir()])

# initialize dataset
dataset = load_dataset("elsaEU/ELSA_D3", split="train", streaming=True)

prompts = {}
for id, row in tqdm(zip(ids, dataset)):
    prompts[id] = row["original_prompt"]  # WARNING: PROMPTS DO NOT MATCH THE IDS!!!

with open("wrong_prompts.json", "w") as f:
    json.dump(prompts, f)
