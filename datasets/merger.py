import json

with open('ru_detection_dataset.json', 'r', encoding='utf-8') as file:
    dataset1 = json.load(file)

with open('small_dataset.json', 'r', encoding='utf-8') as file:
    dataset2 = json.load(file)

max_id = max(item['id'] for item in dataset1)

for item in dataset2:
    item['id'] += max_id

merged_dataset = dataset1 + dataset2

with open('ru_detection_dataset_2.json', 'w', encoding='utf-8') as file:
    json.dump(merged_dataset, file, ensure_ascii=False, indent=4)

print(f"Merged dataset saved to 'merged_dataset.json'.")
