# Baseline Model (BERT for Sequence Classification)
import torch
import os
import json
from datasets import Dataset
from transformers import AutoTokenizer, DistilBertForSequenceClassification, EarlyStoppingCallback, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import f1_score
import argparse
from utils import read_labeled_data, pair_sentences_with_labels

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--eval", type=bool, default=False)
  parser.add_argument("--checkpoint", type=str, default="", help="path to checkpoint to resume training.")

  return parser.parse_args()

# ## Dataset Creation (pairs of sentences & labels)
def create_dataset(dataset_split, difficulty):
  # labels is dict with keys: 'authors', 'changes'
  all_pairs = []
  all_labels = []
  if difficulty == "all":
    easy_probs, easy_labels = read_labeled_data(f"data/easy/{dataset_split}")
    med_probs, med_labels = read_labeled_data(f"data/medium/{dataset_split}")
    hard_probs, hard_labels = read_labeled_data(f"data/hard/{dataset_split}")

    # now create pairs of sentences, in pandas dictionary
    easy_pairs, easy_labels = pair_sentences_with_labels(easy_probs, easy_labels)
    med_pairs, med_labels = pair_sentences_with_labels(med_probs, med_labels)
    hard_pairs, hard_labels = pair_sentences_with_labels(hard_probs, hard_labels)

    all_pairs = easy_pairs + med_pairs + hard_pairs
    all_labels = easy_labels + med_labels + hard_labels
  elif difficulty == "hard":
    hard_probs, hard_labels = read_labeled_data(f"data/hard/{dataset_split}")
    hard_pairs, hard_labels = pair_sentences_with_labels(hard_probs, hard_labels)

    all_pairs = hard_pairs
    all_labels = hard_labels

  assert len(all_pairs) == len(all_labels)
  dataset = Dataset.from_dict({
      "sentence1": [s1 for s1, s2 in all_pairs],
      "sentence2": [s2 for s1, s2 in all_pairs],
      "label": all_labels
  })

  return dataset
  
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Convert logits to predicted class labels
    predictions = logits.argmax(axis=-1)
    # Compute Macro F1 score
    f1 = f1_score(labels, predictions, average='macro')
    return {"f1": f1}

def main(args):
  train_dataset = create_dataset("train", "hard")
  val_dataset = create_dataset("validation", "hard")

  if(args.checkpoint != ""):
    def tokenize_batch(batch):
      return tokenizer(batch['sentence1'], batch['sentence2'], padding=True, return_tensors='pt', truncation=True)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)

    # # Load training arguments
    tokenized_train = train_dataset.map(tokenize_batch, batched=True)
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    tokenized_val = val_dataset.map(tokenize_batch, batched=True)
    tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Load Trainer and continue training
    training_args = TrainingArguments(
      output_dir='./results',          # Output directory for saving model
      eval_strategy="epoch",           # Evaluate after every epoch
      save_strategy="epoch",
      fp16=True,
      learning_rate=2e-5,              # Learning rate
      warmup_steps=500,                # Gradually increase lr at start
      lr_scheduler_type="linear",      # Decreases lr as training progresses
      per_device_train_batch_size=8,   # Batch size for training
      per_device_eval_batch_size=16,   # Batch size for evaluation
      gradient_accumulation_steps=2,   # Effective batch size = batch_size * 2
      num_train_epochs=3,              # Number of epochs
      weight_decay=0.01,               # Strength of weight decay
      logging_dir='./logs',            # Directory for storing logs
      logging_steps=100,               # Log every 100 steps
      load_best_model_at_end=True,     # Load the best model when finished training
    )

    trainer = Trainer(model=model,                         # The model to train
      args=training_args,                  # The training arguments
      train_dataset=tokenized_train,       # The training dataset
      eval_dataset=tokenized_val,          # The evaluation dataset
      tokenizer=tokenizer,                 # The tokenizer
      compute_metrics=compute_metrics,
      callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])

    trainer.train(resume_from_checkpoint=args.checkpoint)
    model.save_pretrained('./finetuned-distilbert')
    tokenizer.save_pretrained('./finetuned-distilbert')

    results = trainer.evaluate()
    print(f"Evaluation Results: {results}")
    pass

  ## Tokenization
  def tokenize_batch(batch):
      return tokenizer(batch['sentence1'], batch['sentence2'], padding=True, return_tensors='pt', truncation=True)

  tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
  model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

  tokenized_train = train_dataset.map(tokenize_batch, batched=True)
  tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

  tokenized_val = val_dataset.map(tokenize_batch, batched=True)
  tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

  print("tokenization over!")

  ## Training
  training_args = TrainingArguments(
    output_dir='./results',          # Output directory for saving model
    eval_strategy="epoch",           # Evaluate after every epoch
    save_strategy="epoch",
    fp16=True,
    learning_rate=2e-5,              # Learning rate
    warmup_steps=500,                # Gradually increase lr at start
    lr_scheduler_type="linear",      # Decreases lr as training progresses
    per_device_train_batch_size=8,   # Batch size for training
    per_device_eval_batch_size=16,   # Batch size for evaluation
    gradient_accumulation_steps=2,   # Effective batch size = batch_size * 2
    num_train_epochs=3,              # Number of epochs
    weight_decay=0.01,               # Strength of weight decay
    logging_dir='./logs',            # Directory for storing logs
    logging_steps=100,               # Log every 100 steps
    load_best_model_at_end=True,     # Load the best model when finished training
  )

  trainer = Trainer(
      model=model,                         # The model to train
      args=training_args,                  # The training arguments
      train_dataset=tokenized_train,       # The training dataset
      eval_dataset=tokenized_val,          # The evaluation dataset
      tokenizer=tokenizer,                 # The tokenizer
      compute_metrics=compute_metrics,
      callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
  )

  trainer.train()

  model.save_pretrained('./finetuned-distilbert')
  tokenizer.save_pretrained('./finetuned-distilbert')

  results = trainer.evaluate()
  print(f"Evaluation Results: {results}")

if __name__ == "__main__":
  args = get_args()
  main(args)