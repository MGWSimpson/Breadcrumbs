from func import run_eng_dataset, run_ru_dataset
from binoculars import Binoculars
import os
import json
import datetime
import argparse

def main():
    parser = argparse.ArgumentParser(description='Запуск тестирования моделей на выбранном датасете')
    parser.add_argument('--dataset', type=str, choices=['ru', 'eng', 'all'], 
                       default='all', help='Выбор датасета для тестирования (ru/eng/all)')
    args = parser.parse_args()

    # Определяем пары моделей для тестирования
    model_pairs = [
        {
            "observer": "tiiuae/Falcon3-7B-Base",
            "performer": "tiiuae/Falcon3-7B-Instruct",
            "name": "Pair 1 - Falcon3-7B-Base and Falcon3-7B-Instruct"
        },
        {
            "observer": "bigscience/bloom-7b1",
            "performer": "strnam/instruction-bloom-7b1",
            "name": "Pair 2 - bigscience/bloom-7b1 and strnam/instruction-bloom-7b1"  
        }
    ]
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)

    for pair in model_pairs:
        print(f"\nTesting {pair['name']}")
        print("-" * 50)
        
        bino = Binoculars(
            mode="accuracy", 
            observer_name_or_path=pair["observer"],
            performer_name_or_path=pair["performer"]
        )

        if args.dataset in ['eng', 'all']:
            results_eng = run_eng_dataset(bino, sample_rate=0.0013, max_samples=2500)
            results_eng['model_pair'] = {
                'observer': pair['observer'],
                'performer': pair['performer'],
                'pair_name': pair['name']
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

            # Обновляем timestamp перед сохранением результатов
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # Заменяем недопустимые символы в имени модели
            model_name = pair['name'].replace(' ', '_').replace('-', '_').replace('/', '_')
            output_file = os.path.join(output_dir, f"results_eng_{model_name}_{timestamp}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results_eng, f, ensure_ascii=False, indent=2)
            print(f"\nResults saved to: {output_file}")

        if args.dataset in ['ru', 'all']:
            results_ru = run_ru_dataset(bino, sample_rate=0.5, max_samples=4000)
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

            print("\nMetrics by dataset:")
            for dataset_name, metrics in results_ru['dataset_metrics'].items():
                if metrics:  # Проверяем, что метрики существуют для датасета
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

            # Обновляем timestamp перед сохранением результатов
            model_name = pair['name'].replace(' ', '_').replace('-', '_').replace('/', '_')
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file_ru = os.path.join(output_dir, f"results_ru_{model_name}_{timestamp}.json")
            with open(output_file_ru, 'w', encoding='utf-8') as f:
                json.dump(results_ru, f, ensure_ascii=False, indent=2)
            print(f"\nResults saved to: {output_file_ru}")

        bino.free_memory()

if __name__ == "__main__":
    main()

    
