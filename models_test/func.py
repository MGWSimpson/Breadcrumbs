from binoculars import Binoculars
import os
import requests
import pyarrow.parquet as pq
import random
import sys
import json
import pandas as pd
from sklearn import metrics
import numpy as np


def run_eng_dataset(bino, sample_rate, max_samples=2000):
    true_positives = []
    false_positives = []
    true_negatives = []
    false_negatives = []
    error_count = 0
    check_counter = 0

    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)

    file_urls = [
        "https://huggingface.co/datasets/artem9k/ai-text-detection-pile/resolve/main/data/train-00000-of-00007-bc5952582e004d67.parquet",
        "https://huggingface.co/datasets/artem9k/ai-text-detection-pile/resolve/main/data/train-00001-of-00007-71c80017bc45f30d.parquet",
        "https://huggingface.co/datasets/artem9k/ai-text-detection-pile/resolve/main/data/train-00002-of-00007-ee2d43f396e78fbc.parquet",
        "https://huggingface.co/datasets/artem9k/ai-text-detection-pile/resolve/main/data/train-00003-of-00007-529931154b42b51d.parquet",
        "https://huggingface.co/datasets/artem9k/ai-text-detection-pile/resolve/main/data/train-00004-of-00007-b269dc49374a2c0b.parquet",
        "https://huggingface.co/datasets/artem9k/ai-text-detection-pile/resolve/main/data/train-00005-of-00007-3dce5e05ddbad789.parquet",
        "https://huggingface.co/datasets/artem9k/ai-text-detection-pile/resolve/main/data/train-00006-of-00007-3d8a471ba0cf1c8d.parquet",
    ]

    filenames = []

    for url in file_urls:
        file_name = url.split("/")[-1]
        local_path = os.path.join(data_dir, file_name)
        filenames.append(local_path)

        if not os.path.exists(local_path):
            print(f"Downloading {file_name}...")
            response = requests.get(url, stream=True)
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"{file_name} download complete!")
        else:
            print(f"{file_name} already exists. Skipping download.")

    print("List of files:")
    print(filenames)

    for filename in filenames:
        parquet_file = pq.ParquetFile(filename)
        print()
        print(f"Loaded {filename}")

        for batch in parquet_file.iter_batches(batch_size=1000):
            if check_counter >= max_samples:
                break
                
            batch_pandas = batch.to_pandas()

            for index, row in batch_pandas.iterrows():
                if random.random() < sample_rate:
                    try:
                        predicted_ai = (bino.predict(row.text) == "Most likely AI-generated")
                    except Exception as e:
                        print(f"Error predicting for text: {row.text}, Error: {e}")
                        error_count += 1
                        continue

                    actually_ai = (row.source == "ai")

                    if predicted_ai and actually_ai:
                        true_positives.append(row.text)
                    elif predicted_ai and not actually_ai:
                        false_positives.append(row.text)
                    elif not predicted_ai and actually_ai:
                        false_negatives.append(row.text)
                    elif not predicted_ai and not actually_ai:
                        true_negatives.append(row.text)

                    check_counter += 1
                    if check_counter % 10 == 0:
                        sys.stdout.write(f"\rProcessed: {check_counter} items")
                        sys.stdout.flush()

    # Преобразование результатов в pandas DataFrame
    results_data = {
        'text': true_positives + false_positives + true_negatives + false_negatives,
        'pred': [1] * len(true_positives) + [1] * len(false_positives) + 
                [0] * len(true_negatives) + [0] * len(false_negatives),
        'class': [1] * len(true_positives) + [0] * len(false_positives) + 
                 [0] * len(true_negatives) + [1] * len(false_negatives)
    }
    score_df = pd.DataFrame(results_data)

    # Вычисление метрик
    f1_score = metrics.f1_score(score_df["class"], score_df["pred"])
    fpr, tpr, thresholds = metrics.roc_curve(y_true=score_df["class"], y_score=score_df["pred"], pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    # Интерполяция TPR при FPR = 0.01%
    tpr_at_fpr_0_01 = np.interp(0.01 / 100, fpr, tpr)

    return {
        'metrics': {
            'f1_score': f1_score,
            'roc_auc': roc_auc,
            'tpr_at_fpr_0_01': tpr_at_fpr_0_01
        },
        'data': {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives,
            'error_count': error_count
        }
    }



def run_ru_dataset(bino, sample_rate, max_samples=2000):
    true_positives = []
    false_positives = []
    true_negatives = []
    false_negatives = []
    error_count = 0
    check_counter = 0

    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)

    json_path = os.path.join(data_dir, "ru_detection_dataset.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Создаем словарь для хранения результатов по датасетам
    dataset_results = {}

    for row in data:
        if check_counter >= max_samples:
            break
            
        if random.random() < sample_rate:
            try:
                predicted_ai = (bino.predict(row["text"]) == "Most likely AI-generated")
            except Exception as e:
                print(f"Error predicting for text: {row['text']}, Error: {e}")
                error_count += 1
                continue

            actually_ai = (row["source"] == "ai")
            dataset_name = row.get("dataset", "unknown")
            example_data = {
                "text": row["text"],
                "dataset": dataset_name,
                "pred": 1 if predicted_ai else 0,
                "class": 1 if actually_ai else 0
            }

            # Инициализируем структуру для нового датасета
            if dataset_name not in dataset_results:
                dataset_results[dataset_name] = {
                    'true_positives': [], 'false_positives': [],
                    'true_negatives': [], 'false_negatives': []
                }

            # Распределяем результаты по датасетам
            if predicted_ai and actually_ai:
                true_positives.append(example_data)
                dataset_results[dataset_name]['true_positives'].append(example_data)
            elif predicted_ai and not actually_ai:
                false_positives.append(example_data)
                dataset_results[dataset_name]['false_positives'].append(example_data)
            elif not predicted_ai and actually_ai:
                false_negatives.append(example_data)
                dataset_results[dataset_name]['false_negatives'].append(example_data)
            elif not predicted_ai and not actually_ai:
                true_negatives.append(example_data)
                dataset_results[dataset_name]['true_negatives'].append(example_data)

            check_counter += 1
            if check_counter % 10 == 0:
                sys.stdout.write(f"\rProcessed: {check_counter} items")
                sys.stdout.flush()

    # Функция для подсчета метрик
    def calculate_metrics(tp, fp, tn, fn):
        if not (tp or fp or tn or fn):
            return None
        
        y_true = ([1] * len(tp) + [0] * len(fp) + 
                 [0] * len(tn) + [1] * len(fn))
        y_pred = ([1] * len(tp) + [1] * len(fp) + 
                 [0] * len(tn) + [0] * len(fn))
        
        if len(set(y_true)) < 2:  # Если есть только один класс
            return {
                'f1_score': None,
                'roc_auc': None,
                'tpr_at_fpr_0_01': None
            }
            
        f1 = metrics.f1_score(y_true, y_pred)
        fpr, tpr, _ = metrics.roc_curve(y_true=y_true, y_score=y_pred, pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)
        tpr_at_fpr_0_01 = np.interp(0.01 / 100, fpr, tpr)
        
        return {
            'f1_score': f1,
            'roc_auc': roc_auc,
            'tpr_at_fpr_0_01': tpr_at_fpr_0_01
        }

    # Подсчет общих метрик
    overall_metrics = calculate_metrics(true_positives, false_positives, 
                                     true_negatives, false_negatives)

    # Подсчет метрик для каждого датасета
    dataset_metrics = {}
    for dataset_name, results in dataset_results.items():
        dataset_metrics[dataset_name] = calculate_metrics(
            results['true_positives'], results['false_positives'],
            results['true_negatives'], results['false_negatives']
        )

    return {
        'overall_metrics': overall_metrics,
        'dataset_metrics': dataset_metrics,
        'data': {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives,
            'error_count': error_count
        }
    }