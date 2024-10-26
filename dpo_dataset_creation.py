from fast_edit_distance import edit_distance
import random
import csv
from tqdm import tqdm
from datasets import load_dataset
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("model/fine_tuned")
tokenizer = AutoTokenizer.from_pretrained("model/fine_tuned")

full_train_ds = load_dataset("grammarly/coedit", split="train")

full_train_ds = full_train_ds.filter(lambda x: x['task'] == 'gec')

full_train_ds = full_train_ds.remove_columns(["_id", "task"])



# Define the dataset generation function outside
def dataset_generation(ds, writer):

    for example in tqdm(ds):

        src_text = example['src']
        tgt_text = example['tgt']

        # Tokenize input
        inputs = tokenizer(src_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

        # Model output with beam search
        try:
            model_output_beam = model.generate(
                **inputs,
                max_length=128,
                min_length=5,
                num_beams=5,
                early_stopping=True,
                do_sample=False,
            )
            pred_text_1 = tokenizer.decode(model_output_beam[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Beam search failed: {e}")
            pred_text_1 = ""

        # Model output with diverse sampling
        try:
            model_output_diverse = model.generate(
                **inputs,
                max_length=128,
                min_length=5,
                num_beams=1,
                do_sample=True,
                top_p=0.9,
                temperature=0.9,
                early_stopping=True,
            )
            pred_text_2 = tokenizer.decode(model_output_diverse[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Diverse generation failed: {e}")
            pred_text_2 = ""

        # Calculate edit distances
        try:
            distance1 = edit_distance(tgt_text, pred_text_1)
        except Exception as e:
            print(f"Error calculating distance for pred1: {e}")
            distance1 = float('inf')  # Assign a large distance if there's an error

        try:
            distance2 = edit_distance(tgt_text, pred_text_2)
        except Exception as e:
            print(f"Error calculating distance for pred2: {e}")
            distance2 = float('inf')  # Assign a large distance if there's an error

        # Compare distances and assign labels
        if distance1 < distance2:
            label1 = 'chosen'
            label2 = 'rejected'
        elif distance1 > distance2:
            label1 = 'rejected'
            label2 = 'chosen'
        else:
            # Equal distances, assign randomly
            label1 = random.choice(["chosen", "rejected"])
            label2 = "rejected" if label1 == "chosen" else "chosen"

        # Append the row to the CSV file
        #print([src_text, pred_text_1, distance1, pred_text_2, distance2, tgt_text])
        writer.writerow([src_text, pred_text_1, label1, pred_text_2, label2, tgt_text])

        file.flush()

    print("Data successfully saved to CSV.")

# File path for the CSV file|
file_path = 'data/full_train_ds_dpo.csv'

# Open the CSV file in append mode (outside of the function)
with open(file_path, mode='a', newline='') as file:
    writer = csv.writer(file)

    # Write headers if the file is empty (only for the first run)
    if file.tell() == 0:
        writer.writerow(['src', 'pred1', 'label1', 'pred2', 'label2', 'tgt'])

    # Pass the writer object to the dataset_generation function
    dataset_generation(full_train_ds, writer)
