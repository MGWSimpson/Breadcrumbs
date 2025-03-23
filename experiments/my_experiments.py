



from binoculars.detector import Binoculars
from binoculars.detector import BINOCULARS_ACCURACY_THRESHOLD as THRESHOLD
from experiments.utils import convert_to_pandas, save_experiment


import os
import argparse
import datetime

import torch
from datasets import Dataset, logging as datasets_logging
import numpy as np
from sklearn import metrics
import pandas as pd

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Args:
    batch_size = 4
    tokens_seen = 512
    job_name= "prompt_included"
    dataset_path = "datasets/robustness/open_orca/default-llama2-13b-chat.jsonl"
    dataset_name = None
    human_sample_key = 'response'
    machine_sample_key = 'meta-llama-Llama-2-13b-chat-hf_generated_text_wo_prompt'
    experiment_path = f"results/{job_name}"
    prompt_key = 'question'
    machine_text_source = 'llama-Llama-2-13b-chat'




 # TODO: https://github.com/collinzrj/output2prompt
 # Utilize the above to invert the prompt presuming this experiment works out.
 

def include_prompt_in_dataset(dataset_path, prompt_key, machine_sample_key, human_sample_key):
    jsonObj = pd.read_json(path_or_buf=dataset_path, lines=True)
    jsonObj[machine_sample_key] = jsonObj[machine_sample_key]
    jsonObj[machine_sample_key + "w_prompt"] = jsonObj[prompt_key] + jsonObj[machine_sample_key]
    jsonObj[human_sample_key] = jsonObj[human_sample_key]
    jsonObj[ human_sample_key + 'w_prompt'] = jsonObj[prompt_key] + jsonObj[human_sample_key]

    jsonObj = jsonObj[:1024]
    return Dataset.from_pandas(jsonObj)


# include the prompt of the one ahead of you.
def include_wrong_prompt_in_dataset(dataset_path, prompt_key, machine_sample_key, human_sample_key):

    jsonObj = pd.read_json(path_or_buf=dataset_path, lines=True)

    for i,j in range(len(jsonObj)):
        if j == len(jsonObj) - 1:
            j = 0

        jsonObj.iloc[i][machine_sample_key] = jsonObj.iloc[j + 1][prompt_key] + jsonObj.iloc[i][machine_sample_key]

    return Dataset.from_pandas(jsonObj)



def compute_metrics_and_save(args, score_df): 
    score_df["pred"] = np.where(score_df["score"] < THRESHOLD, 1, 0)

    # Compute metrics
    f1_score = metrics.f1_score(score_df["class"], score_df["pred"])
    score = -1 * score_df["score"]  # We negative scale the scores to make the class 1 (machine) the positive class
    fpr, tpr, thresholds = metrics.roc_curve(y_true=score_df["class"], y_score=score, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    # Interpolate the TPR at FPR = 0.01%, this is a fixed point in roc curve
    tpr_at_fpr_0_01 = np.interp(0.01 / 100, fpr, tpr)

    # Save experiment
    save_experiment(args, score_df, fpr, tpr, f1_score, roc_auc, tpr_at_fpr_0_01)




def run_experiment(): 
    args = Args()
    os.makedirs(f"{args.experiment_path}", exist_ok=True)

  
    
    
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")

    bino = Binoculars(mode="accuracy", max_token_observed=args.tokens_seen)


    regular_json_obj = pd.read_json(path_or_buf=args.dataset_path, lines=True)
    regular_json_obj = regular_json_obj[:100]
    regular_ds = Dataset.from_pandas(regular_json_obj)

    # breakpoint()
    print(f"Scoring regular machine text")
    machine_regular_scores = regular_ds.map(
        lambda batch: {"score": bino.compute_score(batch[args.machine_sample_key])},
        batched=True,
        batch_size=args.batch_size,
        remove_columns=regular_ds.column_names,
        desc="Scoring regular machine text"
    )

    human_regular_scores = regular_ds.map(
        lambda batch: {"score": bino.compute_score(batch[args. human_sample_key])},
        batched=True,
        batch_size=args.batch_size,
        remove_columns=regular_ds.column_names,
        desc="Scoring regular human text")

    regular_score_df = convert_to_pandas(human_regular_scores, machine_regular_scores)
    
    compute_metrics_and_save(args, regular_score_df)







if __name__ == "__main__": 
    run_experiment()

 