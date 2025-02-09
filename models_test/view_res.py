from func import run_eng_dataset, run_ru_dataset
from binoculars import Binoculars
import os
import json
import datetime

# Определяем пары моделей для тестирования
model_pairs = [
    {
        "observer": "deepseek-ai/deepseek-llm-7b-chat",
        "performer": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        "name": "Pair 1 - deepseek-llm-7b-chat and deepseek-coder-7b-instruct-v1.5"
    },

    {
        "observer": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        "performer": "deepseek-ai/deepseek-llm-7b-chat",
        "name": "Pair 2 - deepseek-coder-7b-instruct-v1.5 and deepseek-llm-7b-chat"
    }
]

# Тестируем каждую пару моделей
for pair in model_pairs:
    print(f"\nTesting {pair['name']}")
    print("-" * 50)
    
    bino = Binoculars(
        mode="accuracy", 
        observer_name_or_path=pair["observer"],
        performer_name_or_path=pair["performer"]
    )

    results_eng = run_eng_dataset(bino, sample_rate=0.001, max_samples=2000)
    
    # Добавляем информацию о моделях в результаты
    results_eng['model_pair'] = {
        'observer': pair['observer'],
        'performer': pair['performer'],
        'pair_name': pair['name']
    }

    # Вывод метрик
    print("English dataset results:")
    print("\nMetrics:")
    print(f"F1 Score: {results_eng['metrics']['f1_score']:.4f}")
    print(f"ROC AUC: {results_eng['metrics']['roc_auc']:.4f}")
    print(f"TPR at 0.01% FPR: {results_eng['metrics']['tpr_at_fpr_0_01']:.4f}")

    print("\nCounts:")
    print(f"True Positives: {len(results_eng['data']['true_positives'])}")
    print(f"False Positives: {len(results_eng['data']['false_positives'])}")
    print(f"True Negatives: {len(results_eng['data']['true_negatives'])}")
    print(f"False Negatives: {len(results_eng['data']['false_negatives'])}")
    print(f"Errors: {results_eng['data']['error_count']}")

    # Сохранение результатов в JSON
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"results_eng_{timestamp}.json")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_eng, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_file}")

    results_ru = run_ru_dataset(bino, sample_rate=0.1, max_samples=2000)
    
    # Добавляем информацию о моделях в результаты
    results_ru['model_pair'] = {
        'observer': pair['observer'],
        'performer': pair['performer'],
        'pair_name': pair['name']
    }

    # Вывод метрик для русского датасета
    print("\nRussian dataset results:")
    print("\nOverall Metrics:")
    print(f"F1 Score: {results_ru['overall_metrics']['f1_score']:.4f}")
    print(f"ROC AUC: {results_ru['overall_metrics']['roc_auc']:.4f}")
    print(f"TPR at 0.01% FPR: {results_ru['overall_metrics']['tpr_at_fpr_0_01']:.4f}")
    """
    print("\nMetrics by dataset:")
    for dataset_name, metrics in results_ru['dataset_metrics'].items():
        if metrics:  # Проверяем, что метрики существуют для датасета
            print(f"\n{dataset_name}:")
            print(f"F1 Score: {metrics['f1_score']:.4f if metrics['f1_score'] is not None else 'N/A'}")
            print(f"ROC AUC: {metrics['roc_auc']:.4f if metrics['roc_auc'] is not None else 'N/A'}")
            print(f"TPR at 0.01% FPR: {metrics['tpr_at_fpr_0_01']:.4f if metrics['tpr_at_fpr_0_01'] is not None else 'N/A'}")
    """
    print("\nCounts:")
    print(f"True Positives: {len(results_ru['data']['true_positives'])}")
    print(f"False Positives: {len(results_ru['data']['false_positives'])}")
    print(f"True Negatives: {len(results_ru['data']['true_negatives'])}")
    print(f"False Negatives: {len(results_ru['data']['false_negatives'])}")
    print(f"Errors: {results_ru['data']['error_count']}")

    # Сохранение результатов в JSON
    output_file_ru = os.path.join(output_dir, f"results_ru_{timestamp}.json")
    with open(output_file_ru, 'w', encoding='utf-8') as f:
        json.dump(results_ru, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_file_ru}")

    bino.free_memory()

    
