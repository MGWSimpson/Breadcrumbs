import json
import os
import glob

def process_json_files(directory):
    bias_data_path = os.path.join(directory, 'bias_data.json')
    try:
        with open(bias_data_path, 'r', encoding='utf-8') as f:
            bias_samples = json.load(f)
        if not isinstance(bias_samples, list):
            raise ValueError("bias_data.json must contain a list of records")
    except Exception as e:
        print(f"Error reading bias_data.json: {str(e)}")
        return

    json_files = [f for f in glob.glob(os.path.join(directory, '*.json')) 
                  if os.path.basename(f) != 'bias_data.json']
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                print(f"Skipping {json_file}: data is not a list")
                continue
                
            original_count = len(data)
            last_id = max(item['id'] for item in data) if data else 0
            
            selected_samples = bias_samples[:original_count]
            
            new_entries = []
            for i, sample in enumerate(selected_samples):
                new_entry = sample.copy()
                new_entry['id'] = last_id + i + 1
                new_entries.append(new_entry)
            
            data.extend(new_entries)
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            print(f"Processed file {json_file}: added {len(new_entries)} records from bias_data")
            
        except Exception as e:
            print(f"Error processing file {json_file}: {str(e)}")

directory = "full_sets/human"
process_json_files(directory)