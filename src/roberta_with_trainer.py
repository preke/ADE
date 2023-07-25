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

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
checkpoint = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


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

    tmp_dict = dataset_dict['train'].train_test_split(test_size=0.2, shuffle=True)
    train_dataset, remaining_dataset = tmp_dict['train'], tmp_dict['test']
    tmp_dict = remaining_dataset.train_test_split(test_size=0.5, shuffle=True)
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
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)




training_args = TrainingArguments(
    'test_trainer',
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 16,
    num_train_epochs =  3,
    learning_rate = 5e-05,
)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

predictions = trainer.predict(tokenized_datasets['validation'])
preds = np.argmax(predictions.predictions, axis=-1)
metric = load_metric('f1')
metric.compute(predictions=preds, references=predictions.label_ids)