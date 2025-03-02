import json

# Список путей к файлам, которые нужно объединить
file_paths = [
    r'E:\diplom\Trinoculars\datasets\summarization-dataset\rew_from_deepseek_r1_dp.json',
    r'E:\diplom\Trinoculars\datasets\summarization-dataset\rew_from_gemini_chat_lp.json',
    r'E:\diplom\Trinoculars\datasets\summarization-dataset\rew_from_o1-mini_two_part.json',
    r'E:\diplom\Trinoculars\datasets\deep_seek_api\new_deep_ans.json'
]

merged_dataset = []
max_id = 0

for file_path in file_paths:
    with open(file_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
        # Обновляем идентификаторы
        for item in dataset:
            item['id'] += max_id
        max_id = max(item['id'] for item in dataset) + 1
        merged_dataset.extend(dataset)

with open(r'E:\diplom\Trinoculars\datasets\ru_detection_dataset_two.json', 'w', encoding='utf-8') as file:
    json.dump(merged_dataset, file, ensure_ascii=False, indent=4)

print("Merged dataset saved to 'ru_detection_dataset_two.json'.")
