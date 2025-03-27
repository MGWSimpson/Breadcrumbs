python run.py \
  --dataset_path ../datasets/core/cc_news/cc_news-llama2_13.jsonl \
  --dataset_name CC-News \
  --human_sample_key text \
  --machine_sample_key meta-llama-Llama-2-13b-hf_generated_text_wo_prompt \
  --machine_text_source LLaMA-2-13B \
  --job_name droput_ablation_1 \
  --dropout_rate 0.05 \
  --n_samples 4

python run.py \
  --dataset_path ../datasets/core/cc_news/cc_news-llama2_13.jsonl \
  --dataset_name CC-News \
  --human_sample_key text \
  --machine_sample_key meta-llama-Llama-2-13b-hf_generated_text_wo_prompt \
  --machine_text_source LLaMA-2-13B \
  --job_name droput_ablation_5 \
  --dropout_rate 0.05 \
  --n_samples 4


  python run.py \
  --dataset_path ../datasets/core/cc_news/cc_news-llama2_13.jsonl \
  --dataset_name CC-News \
  --human_sample_key text \
  --machine_sample_key meta-llama-Llama-2-13b-hf_generated_text_wo_prompt \
  --machine_text_source LLaMA-2-13B \
  --job_name droput_ablation_10 \
  --dropout_rate 0.1 \
  --n_samples 4


  python run.py \
  --dataset_path ../datasets/core/cc_news/cc_news-llama2_13.jsonl \
  --dataset_name CC-News \
  --human_sample_key text \
  --machine_sample_key meta-llama-Llama-2-13b-hf_generated_text_wo_prompt \
  --machine_text_source LLaMA-2-13B \
  --job_name droput_ablation_50 \
  --dropout_rate 0.5 \
  --n_samples 4
