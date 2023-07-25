from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments,Trainer
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, PeftType, PromptEncoderConfig

from datasets import load_dataset
import evaluate
import torch

model_name_or_path = "roberta-large"
task = "mrpc"
num_epochs = 20
lr = 1e-3
batch_size = 32
