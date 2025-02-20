import json
import re
import os
from collections import defaultdict

def sanitize_filename(s):
    return re.sub(r"[^\w\-.]", "_", s)

def split_json_by_dataset(input_filename):
    with open(input_filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    groups = defaultdict(list)
    for record in data:
        dataset = record.get("dataset", "unknown")
        groups[dataset].append(record)
    
    output_directory = "full_sets"
    os.makedirs(output_directory, exist_ok=True)
    
    for dataset_value, records in groups.items():
        for new_id, record in enumerate(records, 1):
            record["id"] = new_id

        safe_name = sanitize_filename(dataset_value)
        output_filename = os.path.join(output_directory, f"{safe_name}.json")
        with open(output_filename, "w", encoding="utf-8") as out_f:
            json.dump(records, out_f, ensure_ascii=False, indent=2)
        print(f"File '{output_filename}' created with {len(records)} records.")

if __name__ == "__main__":
    input_file = "ru_detection_dataset.json"
    split_json_by_dataset(input_file)