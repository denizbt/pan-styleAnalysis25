#!/usr/bin/env python
# coding: utf-8

# # Baseline Model (BERT for Sequence Classification)

import torch
import os
import pandas as pd
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, BertForSequenceClassification
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments


# ## Dataset Creation (pairs of sentences & labels)

## read in data from dir with txt and json 
def read_labeled_data(dir):
  problems = [] # list of lists, each list is a problem and contains the list of sentences
  labels = []
  files = os.listdir(dir)
  files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
  for file in files:
    #print(file)
    with open(dir+"/"+file, "r") as f:
      if(file.endswith(".txt")):
        # a problem file
        problem = f.readlines()
        problems.append([p.strip() for p in problem]) # assumption: split based on periods, not newlines
      else:
        # a truth file
        truth = json.load(f)
        labels.append(truth)
  
  assert len(problems) == len(labels)
  return problems, labels


def pair_sentences_with_labels(problems, labels):
    sentence_pairs = []
    label_pairs = []

    count = 1
    for prob, label in zip(problems, labels):
        # For each problem (list of sentences) and corresponding label (dict with 'changes' list)
        changes = label['changes']

        if len(prob) - 1 != len(changes):
            continue # hard training problem-3207.txt is broken (extra new line I think, the number of sentence pairs does not match)
            # raise ValueError(f"Number of sentence pairs does not match the number of labels for this file. "
            #                  f"Sentences: {len(prob) - 1}, Changes: {len(changes)}")
        
        for i in range(len(prob) - 1):
            sentence_pairs.append((prob[i], prob[i + 1]))
            label_pairs.append(changes[i])
        
        count += 1

    return sentence_pairs, label_pairs


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
  

# from transformers import AdamW, get_scheduler
# optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

# train_dataloader = DataLoader(tokenized_dataset, batch_size=8, shuffle=True)
# num_training_steps = len(train_dataloader) * 3  # Assume 3 epochs
# lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

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

  print("tokenization over!") # works up to here!

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