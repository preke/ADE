import json
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict, Features, ClassLabel, Value

from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import Trainer

from datasets import load_metric
import numpy as np


from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PromptEncoderConfig,
)
from datasets import load_dataset
import evaluate
import torch

model_name_or_path = "roberta-base"

SEED               = 42



def load_data(tsv_file):
    df = pd.read_csv(tsv_file, sep='\t')
    data_path = '../data/tmp.jsonl'
    json_data = df[['sent', 'labels']].to_dict(orient="records")
    with open(data_path, 'w') as outfile:
        for row in json_data:
            json.dump(row, outfile)
            outfile.write('\n')

    class_names = ['negative', 'positive']
    features = Features({'sent': Value('string'), 'labels': ClassLabel(names=class_names)})

    dataset_dict = load_dataset("json", data_files=data_path, features=features)

    tmp_dict = dataset_dict['train'].train_test_split(test_size=0.2, shuffle=True, seed=SEED)
    train_dataset, remaining_dataset = tmp_dict['train'], tmp_dict['test']
    tmp_dict = remaining_dataset.train_test_split(test_size=0.5, shuffle=True, seed=SEED)
    valid_dataset, test_dataset = tmp_dict['train'], tmp_dict['test']
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': valid_dataset,
        'test': test_dataset
    })
    return dataset_dict

def tokenize_function(example):
    return tokenizer(example["sent"], truncation=True)


friends_persona_a = '../data/Friends_A.tsv'

dataset_dict = load_data(friends_persona_a)
tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)







trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()


