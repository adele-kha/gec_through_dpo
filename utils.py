import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate
from tqdm import tqdm

def create_tokenize_function(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # Tokenizer initialized here only once

    def tokenize_function(examples):
        # Use the initialized tokenizer
        model_inputs = tokenizer(
            examples['src'],  # Use the source sentence as input
            max_length=128,
            truncation=True,
            padding="max_length"
        )

        # Tokenize the target (corrected sentence) to create the labels
        with tokenizer.as_target_tokenizer():  # Ensures labels are handled correctly
            labels = tokenizer(
                examples['tgt'],  # Use the target sentence for labels
                max_length=128,
                truncation=True,
                padding="max_length"
            )

        # Process labels: convert padding tokens to -100 for loss calculation
        labels = labels["input_ids"]
        labels = [[-100 if token == tokenizer.pad_token_id else token for token in label] for label in labels]

        # Add labels to the model inputs
        model_inputs["labels"] = labels

        return model_inputs

    return tokenize_function



def evaluate_model(model, tokenizer, ds):
  preds = []
  targets = []

  for example in tqdm(ds):
    src_text = example['src']
    tgt_text = example['tgt']

    inputs = tokenizer(src_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.0)

    pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    #print(f"Input: {src_text}")
    #print(f"Prediction: {pred_text}")
    #print(f"Target: {tgt_text}\n")
    preds.append(pred_text)
    targets.append([tgt_text])

    if inputs != pred_text:
      print(f"Input: {src_text}")
      print(f"Prediction: {pred_text}")
      print(f"Target: {tgt_text}\n")


  bleu = evaluate.load("bleu")


  results = bleu.compute(predictions=preds, references=targets)
  return results["bleu"]

def select_chosen_pred(example):
    if example['label1'] == 'chosen':
        return {'pred': example['pred1'], 'tgt': example['tgt']}
    else:
        return {'pred': example['pred2'], 'tgt': example['tgt']}