import os
from trl import DPOConfig, DPOTrainer
from datasets import Dataset, load_dataset
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments

from transformers import AdamW
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader

file_path = 'data/full_train_ds_dpo.csv'

dataset = load_dataset('csv', data_files=file_path)

model_name = "t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

tokenizer.pad_token = tokenizer.eos_token  
model.config.pad_token_id = tokenizer.pad_token_id  

def select_chosen_pred(example):
    if example['label1'] == 'rejected':
        return {'anchor': example['src'], 
                'positive': example['pred2'],
                'negative': example['pred1']}
    else:
        return {'anchor': example['src'], 
                'positive': example['pred1'],
                'negative': example['pred2']}
    
new_dataset = dataset.map(select_chosen_pred)
new_dataset = new_dataset.remove_columns(["src","pred1","label1","pred2","label2","tgt"])

max_length = 128

triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-5)

epochs = 3

def train_contrastive(model, tokenizer, dataset, epochs=3, batch_size=8, max_length=128):
    model.train()  
    
    for epoch in range(epochs):
        total_loss = 0
        print(f"Epoch {epoch + 1}/{epochs}")

        train_loader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)  
        
        for i, batch in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()  

            src_text = batch['anchor']
            tgt_text = batch['positive']
            incorrect_text = batch['negative']

            src_enc = tokenizer(src_text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(device)
            tgt_enc = tokenizer(tgt_text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(device)
            incorrect_enc = tokenizer(incorrect_text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(device)

            src_rep = model.encoder(**src_enc).last_hidden_state[:, 0, :]  
            tgt_rep = model.encoder(**tgt_enc).last_hidden_state[:, 0, :]
            incorrect_rep = model.encoder(**incorrect_enc).last_hidden_state[:, 0, :]
            
            loss = triplet_loss_fn(src_rep, tgt_rep, incorrect_rep)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}")
        
    print("Training complete!")


output_dir = 'models/experimental'

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model saved to {output_dir}")