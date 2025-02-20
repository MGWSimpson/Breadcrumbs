import time
import random
from openai import OpenAI
import os
import json
from tqdm import tqdm
from datasets import load_dataset

ds = load_dataset("Vikhrmodels/GrandMaster-PRO-MAX")
train_ds = ds['train']

var_name = os.getenv("OPENAI_API_KEY")
client = OpenAI()
OpenAI.api_key = var_name

def generate_paragraph(model="gpt-4o-mini", max_tokens=1000, temperature=0.7):
    while (True):
        random_index = random.randint(0, len(train_ds) - 1)
        conv1 = train_ds[random_index]
        
        if conv1['prompt_lang'] == 'ru':
            conv2 = conv1['conversation'][0]
            break
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Ты — помощник, который даёт развернутые ответы на русском языке. Пиши одним абзацем без лишних символов и перечислений."},
            conv2
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        n=1
    )
    answer = response.choices[0].message.content.strip()
    return answer

def create_dataset(num_paragraphs=10, output_file="dataset.json"):
    dataset = []
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"Found existing dataset with {len(dataset)} entries")
    except FileNotFoundError:
        print("Creating new dataset")
    
    start_id = len(dataset) + 1
    
    for i in tqdm(range(num_paragraphs), desc="Generating paragraphs", unit="paragraph"):
        text = generate_paragraph()
        
        entry = {
            "id": start_id + i,
            "text": text,
            "source": "ai",
            "dataset": "chatGPT 4o-mini prompts from Vikhrmodels/GrandMaster-PRO-MAX"
        }
        dataset.append(entry)
        
        time.sleep(0.01)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    
    print(f"Done! Added {num_paragraphs} new entries to '{output_file}'.")

if __name__ == "__main__":
    create_dataset(num_paragraphs=10, output_file="ru_detection_dataset.json")
