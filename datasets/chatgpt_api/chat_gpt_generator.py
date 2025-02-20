import time
import random
from openai import OpenAI
import os
import json
from tqdm import tqdm

var_name = os.getenv("OPENAI_API_KEY")

client = OpenAI()
OpenAI.api_key = var_name

def generate_paragraph(prompt, model="gpt-4o-mini", max_tokens=1000, temperature=0.9):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Ты — помощник, который даёт развернутые ответы на русском языке. Пиши одним абзацем без лишних символов и перечислений."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        n=1
    )
    answer = response.choices[0].message.content.strip()
    return answer

def create_dataset(num_paragraphs=10, output_file="dataset.json"):
    with open('prompts.json', 'r', encoding='utf-8') as f:
        prompts = json.load(f)['prompts']
    
    dataset = []
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"Found existing dataset with {len(dataset)} entries")
    except FileNotFoundError:
        print("Creating new dataset")
    
    start_id = len(dataset) + 1
    
    for i in tqdm(range(num_paragraphs), desc="Generating paragraphs", unit="paragraph"):
        prompt = random.choice(prompts)
        text = generate_paragraph(prompt)
        
        entry = {
            "id": start_id + i,
            "text": text,
            "source": "ai",
            "dataset": "chatGPT 4o-mini"
        }
        dataset.append(entry)
        
        time.sleep(0.01)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    
    print(f"Done! Added {num_paragraphs} new entries to '{output_file}'.")

if __name__ == "__main__":
    create_dataset(num_paragraphs=200, output_file="small_dataset.json")
