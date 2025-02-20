import time
import random
import os
import json
from tqdm import tqdm

from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="",
)

def generate_paragraph(prompt, temperature=0.75):
    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "<YOUR_SITE_URL>",

            "X-Title": "<YOUR_SITE_NAME>",
        },
        model="deepseek/deepseek-r1:free",
        messages=[
            {"role": "system", "content": "Ты — помощник, который даёт развернутые ответы на русском языке. Пиши одним абзацем без лишних символов и перечислений."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        )
        answer = completion.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print(f"Error generating paragraph: {e}")
        time.sleep(120)
        return ""


def append_entry_to_json_array(entry, output_file):
    """
    Adds a new entry to the JSON array stored in output_file, maintaining formatting.
    Creates a new JSON array if the file doesn't exist.
    """
    indent = " " * 4
    if not os.path.exists(output_file):
        with open(output_file, "w", encoding="utf-8") as f:
            entry_str = json.dumps(entry, ensure_ascii=False, indent=4)
            entry_lines = entry_str.splitlines()
            # Добавляем внешний отступ для записи внутри массива
            indented_entry = "\n".join(indent + line for line in entry_lines)
            f.write("[\n" + indented_entry + "\n]\n")
        return

    # Если файл уже существует, открываем его для чтения и обновления
    with open(output_file, "r+", encoding="utf-8") as f:
        content = f.read().rstrip()
        pos = content.rfind(']')
        if pos == -1:
            raise ValueError("Invalid file format: missing closing bracket")
        # Определяем, есть ли уже записи в массиве (между '[' и ']')
        inner_content = content[content.find('[') + 1:pos].strip()
        comma = "," if inner_content else ""
        entry_str = json.dumps(entry, ensure_ascii=False, indent=4)
        entry_lines = entry_str.splitlines()
        indented_entry = "\n".join(indent + line for line in entry_lines)
        # Формируем новое содержимое файла: часть до закрывающей скобки, запятая (если нужны), новая запись, и затем закрывающая скобка
        new_content = content[:pos].rstrip() + comma + "\n" + indented_entry + "\n" + content[pos:]
        f.seek(0)
        f.truncate()
        f.write(new_content)

def create_dataset(num_paragraphs=10, output_file="dataset.json"):
    # Чтение подсказок из файла prompts.json
    with open('prompts.json', 'r', encoding='utf-8') as f:
        prompts = json.load(f)['prompts']
    
    # Определяем стартовый id (последовательное нумерование)
    if os.path.exists(output_file):
        # Загружаем существующий датасет для определения количества записей
        with open(output_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                start_id = len(data) + 1
            except json.JSONDecodeError:
                start_id = 1
        print(f"Found existing dataset with {start_id - 1} entries")
    else:
        print("Creating new dataset")
        start_id = 1

    # Генерация и запись новых записей по одной
    for i in tqdm(range(num_paragraphs), desc="Generating paragraphs", unit="paragraph"):
        prompt = random.choice(prompts)
        text = generate_paragraph(prompt)
        
        # Если текст пустой, пропускаем запись
        if not text.strip():
            print("Empty response received, skipping entry.")
            continue
        
        entry = {
            "id": start_id + i,
            "text": text,
            "source": "ai",
            "dataset": "deepseek-r1"
        }
        # Записываем именно новую запись, а не весь датасет целиком
        append_entry_to_json_array(entry, output_file)

        time.sleep(60)
    
    print(f"Done! Added {num_paragraphs} new entries to '{output_file}'.")

if __name__ == "__main__":
    create_dataset(num_paragraphs=1000, output_file="small_dataset_2.json")
