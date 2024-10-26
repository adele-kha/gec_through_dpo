import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--MODEL_DIR', type=str, required=True, help='Model dir')

args = parser.parse_args()

# Load the model and tokenizer from the directory
model = AutoModelForSeq2SeqLM.from_pretrained(args.MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(args.MODEL_DIR)

test = 'Fix grammatically: I likes turtles.'

# Now you can use the model for inference
inputs = tokenizer(test, return_tensors="pt")
outputs = model.generate(**inputs)

# Decode the outputs
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("test:", test)
print("corrected:", generated_text)
