import datasets
from datasets import load_dataset, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import evaluate
from tqdm import tqdm
import argparse
from utils import evaluate_model

parser = argparse.ArgumentParser()

parser.add_argument('--MODEL DIR', type=str, required=True, help='Model dir')

args = parser.parse_args()

model = AutoModelForSeq2SeqLM.from_pretrained(args.MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(args.MODEL_DIR)

full_test_ds = load_dataset("grammarly/coedit", split="validation")
full_test_ds = full_test_ds.filter(lambda x: x['task'] == 'gec')
full_test_ds = full_test_ds.remove_columns(["_id", "task"])

print(evaluate_model(model, tokenizer, full_test_ds))


