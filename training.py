print("foo")

import datasets

print("foo")

from datasets import load_dataset
from datasets import Dataset

print("foo")

import os
import trl
import torch
torch.device('cpu')
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from trl import SFTConfig, SFTTrainer

from utils import create_tokenize_function, select_chosen_pred

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--BASE_MODEL', type=str, required=True, help='Base model name')
parser.add_argument('--EPOCHS', type=int, default=3, help='Number of epochs')
parser.add_argument('--LEARNING_RATE', type=float, default=1e-5, help='Learning rate for training')

args = parser.parse_args()

print("foo")

if args.BASE_MODEL=="t5-small":
    full_ds = load_dataset("grammarly/coedit", split="train")

    full_ds = full_ds.filter(lambda x: x['task'] == 'gec')

    full_ds = full_ds.remove_columns(["_id", "task"])

    train_valid_split = full_ds.train_test_split(test_size=0.2)
    
    full_train_ds = train_valid_split['train']
    full_eval_ds = train_valid_split['test']

    output_dir = 'models/fine_tuned'

elif args.BASE_MODEL=='model/fine_tuned':
    full_train_ds = load_dataset('csv', data_files='data/full_train_ds_dpo.csv')
    full_train_ds = full_train_ds.remove_columns(['src'])
    full_train_ds = full_train_ds.map(select_chosen_pred)
    full_train_ds = full_train_ds.remove_columns(['pred1', 'label1', 'pred2', 'label2'])
    full_train_ds = full_train_ds.rename_column('pred', 'src')
    full_train_ds = full_train_ds['train']

    train_valid_split = full_train_ds.train_test_split(test_size=0.2)
    
    full_train_ds = train_valid_split['train']
    full_eval_ds = train_valid_split['test']

    output_dir = 'models/dpo'

print("Data loaded")

model_name = args.BASE_MODEL

tokenizer = AutoTokenizer.from_pretrained(model_name)

 

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

tokenizer.pad_token = tokenizer.eos_token  
model.config.pad_token_id = tokenizer.pad_token_id  

tokenize_fn = create_tokenize_function(model_name)

tokenized_train_dataset = full_train_ds.map(tokenize_fn, batched=True)
tokenized_eval_dataset = full_eval_ds.map(tokenize_fn, batched=True)

print("tokenized", tokenized_train_dataset[0])

tokenized_train_dataset = tokenized_train_dataset.remove_columns(['src', 'tgt'])
tokenized_eval_dataset = tokenized_eval_dataset.remove_columns(['src', 'tgt'])

print("Finished tokenization")



training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    num_train_epochs=3,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,

)

trainer.train()


model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model saved to {output_dir}")


