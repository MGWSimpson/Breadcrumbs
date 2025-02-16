from func import run_eng_dataset, run_ru_dataset
from binoculars import Trinoculars
import os
import json
import datetime
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='Запуск тестирования моделей на выбранном датасете с использованием Trinoculars'
    )
    parser.add_argument(
        '--dataset', type=str, choices=['ru', 'eng', 'all'],
        default='all', help='Выбор датасета для тестирования (ru/eng/all)'
    )
    args = parser.parse_args()

    # Определяем тройки моделей для тестирования
    model_triples = [
        {
            "model1": "deepseek-ai/deepseek-llm-7b-base",
            "model2": "deepseek-ai/deepseek-llm-7b-chat",
            # Для примера используем ту же модель, что и observer, чтобы токенизаторы были совместимы
            "model3": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
            "name": "Triple 1 - deepseek-llm-7b-base, chat, deepseek-coder-7b-instruct-v1.5"
        }
    ]

    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)

    for triple in model_triples:
        print(f"\nTesting {triple['name']}")
        print("-" * 50)

        # Инициализируем новый объект Trinoculars с передачей трёх моделей
        trinoculars = Trinoculars(
            model1_name_or_path=triple["model1"],
            model2_name_or_path=triple["model2"],
            model3_name_or_path=triple["model3"]
        )

        if args.dataset in ['eng', 'all']:
            results_eng = run_eng_dataset(trinoculars, sample_rate=0.0013, max_samples=2500)
            results_eng['model_triple'] = {
                'model1': triple['model1'],
                'model2': triple['model2'],
                'model3': triple['model3'],
                'name': triple['name']
            }
            
            # Вывод метрик для английского датасета
            print("\nEnglish dataset results:")
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

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = triple['name'].replace(' ', '_').replace('-', '_').replace('/', '_')
            output_file = os.path.join(output_dir, f"results_eng_{model_name}_{timestamp}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results_eng, f, ensure_ascii=False, indent=2)
            print(f"\nResults saved to: {output_file}")

        if args.dataset in ['ru', 'all']:
            results_ru = run_ru_dataset(trinoculars, sample_rate=0.5, max_samples=4000)
            results_ru['model_triple'] = {
                'model1': triple['model1'],
                'model2': triple['model2'],
                'model3': triple['model3'],
                'name': triple['name']
            }
            
            # Вывод метрик для русского датасета
            print("\nRussian dataset results:")
            print("\nOverall Metrics:")
            print(f"F1 Score: {results_ru['overall_metrics']['f1_score']:.4f}")
            print(f"ROC AUC: {results_ru['overall_metrics']['roc_auc']:.4f}")
            print(f"TPR at 0.01% FPR: {results_ru['overall_metrics']['tpr_at_fpr_0_01']:.4f}")

            print("\nMetrics by dataset:")
            for dataset_name, metrics in results_ru['dataset_metrics'].items():
                if metrics:
                    print(f"\n{dataset_name}:")
                    f1_score = f"{metrics['f1_score']:.4f}" if isinstance(metrics['f1_score'], (int, float)) else "N/A"
                    roc_auc = f"{metrics['roc_auc']:.4f}" if isinstance(metrics['roc_auc'], (int, float)) else "N/A"
                    tpr = f"{metrics['tpr_at_fpr_0_01']:.4f}" if isinstance(metrics['tpr_at_fpr_0_01'], (int, float)) else "N/A"
                    print(f"F1 Score: {f1_score}")
                    print(f"ROC AUC: {roc_auc}")
                    print(f"TPR at 0.01% FPR: {tpr}")

            print("\nCounts:")
            print(f"True Positives: {len(results_ru['data']['true_positives'])}")
            print(f"False Positives: {len(results_ru['data']['false_positives'])}")
            print(f"True Negatives: {len(results_ru['data']['true_negatives'])}")
            print(f"False Negatives: {len(results_ru['data']['false_negatives'])}")
            print(f"Errors: {results_ru['data']['error_count']}")

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = triple['name'].replace(' ', '_').replace('-', '_').replace('/', '_')
            output_file_ru = os.path.join(output_dir, f"results_ru_{model_name}_{timestamp}.json")
            with open(output_file_ru, 'w', encoding='utf-8') as f:
                json.dump(results_ru, f, ensure_ascii=False, indent=2)
            print(f"\nResults saved to: {output_file_ru}")

        # Освобождаем память, выгружая модели
        trinoculars.free_memory()

if __name__ == "__main__":
    main()

    
