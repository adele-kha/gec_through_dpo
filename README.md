# Fine-Tuning T5-Small for Grammatical Error Correction (GEC) Using Grammarly CoEdit Dataset

This project focuses on fine-tuning the t5-small model for grammatical error correction (GEC) with a multi-step training approach:


Stage 1: Fine-tuning the pre-trained T5 model on the Grammarly CoEdit dataset, which contains sentence pairs of grammatically incorrect (source) and corrected (target) sentences.

Stage 2 (DPO): A dataset is created using edit distance to choose between two corrected sentences, and the fine-tuned model is further trained using Direct Preference Optimization (DPO). The sentence that is closer in distance to the target is chosen as the new source sentence. 

Alternative Stage 2 (Contrastive loss): Instead of using DPO, we fine-tune the model using a contrastive loss function (triplet loss) with the same dataset. In this variant, the sentence that is further away from the target is treated as a negative example and the sentence that is closer as positive.

Each time for evaluation, we use BLEU. BLEU (Bilingual Evaluation Understudy) is a metric used to evaluate the quality of machine-generated text, particularly in tasks like machine translation and text generation. It measures how similar the generated text is to one or more reference texts by comparing the n-grams (contiguous sequences of n items) in the generated output to those in the reference texts.

Before fine-tuning, BLEU was equal to 0.31. After the first round of fine-tuning, it reached 0.47, and after the second round -- 0.48. In contrast, the experimental approach failed to show good enough results (0.24). 

It is understandable why cross-entropy is superior to contrastive loss when fine-tuning a model for GEC. Contrastive learning often relies on comparing a pair of outputs (e.g., one correct and one incorrect) and may not provide the fine-grained, token-level feedback required for grammar corrections. In contrast, traditional supervised learning with cross-entropy loss gives direct feedback on each token's correctness, which is crucial for tasks that require precision.

to fine-tune:

python training.py --BASE_MODEL t5-small

to test a single instance after fine-tuning:

python testing.py --MODEL_DIR models/fine_tuned

run full_evalution:

python evaluation.py --MODEL DIR models/fine_tuned

to create dpo dataset:

python dpo_dataset_creation.py

to further fine-tune based on the dpo dataset:

python training.py --BASE_MODEL models/fine_tuned

full_evaluation:

python evaluation.py --MODEL DIR models/dpo

create an experimental model with contrastive learning:

python experimental.py

evaluate:
python evaluation.py --MODEL DIR models/experimental

