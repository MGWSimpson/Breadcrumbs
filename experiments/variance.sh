python run.py \
  --dataset_path ../datasets/core/cc_news/cc_news-llama2_13.jsonl \
  --dataset_name CC-News \
  --human_sample_key text \
  --machine_sample_key meta-llama-Llama-2-13b-hf_generated_text_wo_prompt \
  --machine_text_source LLaMA-2-13B \ 
  --n_samples 5 \
  --seed 0
  --job_name variance_experiment_5_0


python run.py \
  --dataset_path ../datasets/core/cc_news/cc_news-llama2_13.jsonl \
  --dataset_name CC-News \
  --human_sample_key text \
  --machine_sample_key meta-llama-Llama-2-13b-hf_generated_text_wo_prompt \
  --machine_text_source LLaMA-2-13B \ 
  --n_samples 5 \
  --seed 1 \
  --job_name variance_experiment_5_1



  python run.py \
  --dataset_path ../datasets/core/cc_news/cc_news-llama2_13.jsonl \
  --dataset_name CC-News \
  --human_sample_key text \
  --machine_sample_key meta-llama-Llama-2-13b-hf_generated_text_wo_prompt \
  --machine_text_source LLaMA-2-13B \ 
  --n_samples 10 \
  --seed 0
  --job_name variance_experiment_10_0


python run.py \
  --dataset_path ../datasets/core/cc_news/cc_news-llama2_13.jsonl \
  --dataset_name CC-News \
  --human_sample_key text \
  --machine_sample_key meta-llama-Llama-2-13b-hf_generated_text_wo_prompt \
  --machine_text_source LLaMA-2-13B \ 
  --n_samples 10 \
  --seed 1 \
  --job_name variance_experiment_10_1


  