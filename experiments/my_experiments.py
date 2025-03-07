



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
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

class Args:
    batch_size = 32
    tokens_seen = 512
    job_name= "test"
    dataset_path = "/content/Binoculars/datasets/robustness/open_orca/default-llama2-13b-chat.jsonl"
    dataset_name = None
    human_sample_key = 'response'
    machine_sample_key = 'meta-llama-Llama-2-13b-chat-hf_generated_text_wo_prompt'
    machine_text_source = None
    prompt_key = 'question'




 # TODO: https://github.com/collinzrj/output2prompt
 # Utilize the above to invert the prompt presuming this experiment works out.
 

def include_prompt_in_dataset(dataset_path, prompt_key, machine_sample_key, human_sample_key):
    jsonObj = pd.read_json(path_or_buf=dataset_path, lines=True)
    jsonObj[machine_sample_key] = jsonObj[prompt_key] + jsonObj[machine_sample_key]
    jsonObj[human_sample_key] = jsonObj[prompt_key] + jsonObj[machine_sample_key]
    

    return Dataset.from_pandas(jsonObj)


# include the prompt of the one ahead of you.
def include_wrong_prompt_in_dataset(dataset_path, prompt_key, machine_sample_key, human_sample_key):

    jsonObj = pd.read_json(path_or_buf=dataset_path, lines=True)

    for i,j in range(len(jsonObj)):
        if j == len(jsonObj) - 1:
            j = 0

        jsonObj.iloc[i][machine_sample_key] = jsonObj.iloc[j + 1][prompt_key] + jsonObj.iloc[i][machine_sample_key]

    return Dataset.from_pandas(jsonObj)



def prompt_inclusion_experiment():
    args = Args()
    print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")

    print(torch.cuda.current_device())
    
    
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"GPU Type: {torch.cuda.get_device_name(1)}")

    bino = Binoculars(mode="accuracy", max_token_observed=args.tokens_seen)

    added_prompt_ds = include_prompt_in_dataset(args.dataset_path, args.prompt_key, args.machine_sample_key, args.human_sample_key)

    
    jsonObj = jsonObj[:5]


    
    regular_json_obj = pd.read_json(path_or_buf=args.dataset_path, lines=True)
    regular_json_obj = regular_json_obj[:5]
    regular_ds = Dataset.from_pandas(regular_json_obj)

    print(f"Scoring added prompt machine text")
    machine_added_prompt_scores = added_prompt_ds.map(
        lambda batch: {"score": bino.compute_score(batch[args.machine_sample_key])},
        batched=True,
        batch_size=args.batch_size,
        remove_columns=added_prompt_ds.column_names
    )
    # TODO: add the scores for the humans too.


    print(f"Scoring regular machine text")
    machine_regular_scores = regular_ds.map(
        lambda batch: {"score": bino.compute_score(batch[args.machine_sample_key])},
        batched=True,
        batch_size=args.batch_size,
        remove_columns=regular_ds.column_names

    )

    score_df = convert_to_pandas(machine_added_prompt_scores, machine_regular_scores)
    score_df.to_csv('score_df.csv')

    # TODO: add the experiment saves for 

def wrong_prompt_inclusion_experiment(): 
    pass




if __name__ == "__main__": 
    prompt_inclusion_experiment()

 