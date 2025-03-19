# Baseline Model (BERT for Sequence Classification)

import torch
import os
import pandas as pd
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, BertForSequenceClassification
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments

# utils.py
from utils import read_labeled_data, pair_sentences_with_labels


# ## Dataset Creation (pairs of sentences & labels)
def create_dataset(dataset_split):
  # labels is dict with keys: 'authors', 'changes'
  easy_probs, easy_labels = read_labeled_data(f"data/easy/{dataset_split}")
  med_probs, med_labels = read_labeled_data(f"data/medium/{dataset_split}")
  hard_probs, hard_labels = read_labeled_data(f"data/hard/{dataset_split}")

  # now create pairs of sentences, in pandas dictionary
  easy_pairs, easy_labels = pair_sentences_with_labels(easy_probs, easy_labels)
  med_pairs, med_labels = pair_sentences_with_labels(med_probs, med_labels)
  hard_pairs, hard_labels = pair_sentences_with_labels(hard_probs, hard_labels)

  all_pairs = easy_pairs + med_pairs + hard_pairs
  all_labels = easy_labels + med_labels + hard_labels
  assert len(all_pairs) == len(all_labels)

  dataset = Dataset.from_dict({
      "sentence1": [s1 for s1, s2 in all_pairs],
      "sentence2": [s2 for s1, s2 in all_pairs],
      "label": all_labels
  })

  return dataset
  
def main():
  train_dataset = create_dataset("train")
  val_dataset = create_dataset("validation")

  ## Tokenization
  tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
  model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

  def tokenize_batch(batch):
    return tokenizer(batch['sentence1'], batch['sentence2'], padding=True, truncation=True)

  tokenized_train = train_dataset.map(tokenize_batch, batched=True)
  tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

  tokenized_val = val_dataset.map(tokenize_batch, batched=True)
  tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
  print("tokenization over!")

  training_args = TrainingArguments(
    output_dir='./results',          # Output directory for saving model
    evaluation_strategy="epoch",     # Evaluate after every epoch
    learning_rate=2e-5,              # Learning rate
    per_device_train_batch_size=8,   # Batch size for training
    per_device_eval_batch_size=16,   # Batch size for evaluation
    num_train_epochs=3,              # Number of epochs
    weight_decay=0.01,               # Strength of weight decay
    logging_dir='./logs',            # Directory for storing logs
    logging_steps=10,                # Log every 10 steps
    load_best_model_at_end=True,     # Load the best model when finished training
)

  trainer = Trainer(
      model=model,                         # The model to train
      args=training_args,                  # The training arguments
      train_dataset=train_dataset,         # The training dataset
      eval_dataset=val_dataset,           # The evaluation dataset
      tokenizer=tokenizer,                 # The tokenizer
  )

  trainer.train()
  model.save_pretrained('./fine_tuned_bert')
  tokenizer.save_pretrained('./fine_tuned_bert')


if __name__ == "__main__":
  main()