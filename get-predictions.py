# get predictions from HuggingFace model checkpoint
import torch
import os
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import argparse
from sklearn.metrics import f1_score
from tqdm import tqdm

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--eval", type=bool, default=False)
  parser.add_argument("--checkpoint", type=str, default="", help="path to checkpoint to resume training.")

  return parser.parse_args()

# Dataset Creation (pairs of sentences & labels)
## read in data from dir with txt and json 
def read_labeled_data(dir):
  problems = [] # list of lists, each list is a problem and contains the list of sentences
  labels = []
  files = os.listdir(dir)
  files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
  for file in files:
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


def create_dataset(dataset_split, difficulty="all") -> list[tuple]:
  # labels is dict with keys: 'authors', 'changes'
  all_labels = []
  all_pairs = []
  if difficulty == "all":
    easy_probs, easy_labels = read_labeled_data(f"data/easy/{dataset_split}")
    med_probs, med_labels = read_labeled_data(f"data/medium/{dataset_split}")
    hard_probs, hard_labels = read_labeled_data(f"data/hard/{dataset_split}")

    # now create pairs of sentences, in pandas dictionary
    easy_pairs, easy_labels = pair_sentences_with_labels(easy_probs, easy_labels)
    med_pairs, med_labels = pair_sentences_with_labels(med_probs, med_labels)
    hard_pairs, hard_labels = pair_sentences_with_labels(hard_probs, hard_labels)

    all_labels = easy_labels + med_labels + hard_labels
    all_pairs = easy_pairs + med_pairs + hard_pairs
  else:
    probs, labels = read_labeled_data(f"data/{difficulty}/{dataset_split}")
    pairs, labels = pair_sentences_with_labels(probs, labels)

    all_labels = labels
    all_pairs = pairs

  return all_pairs, all_labels
  
def predict_by_batch(args, sentence_pairs, batch_size=16):
    all_predictions = []

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Loop through the data in batches
    for i in tqdm(range(0, len(sentence_pairs), batch_size)):
        batch = sentence_pairs[i:i + batch_size]  # get a batch

        inputs = tokenizer.batch_encode_plus(
            batch,
            padding=True,        # Pads sequences to the longest one
            truncation=True,     # Truncates sequences to the max length
            return_tensors="pt"
        )

        # Run inference for this batch
        with torch.no_grad():
            outputs = model(**inputs)

        # For binary classification with two outputs (num_labels=2)
        predictions = torch.argmax(outputs.logits, dim=-1).numpy()

        # add predictions from this batch to total preds
        all_predictions.extend(predictions)

    return all_predictions

def main(args):
  difficulty = "medium"
  sentence_pairs, labels = create_dataset("validation", difficulty)
  print("read the data!")

  predictions = predict_by_batch(args, sentence_pairs)
  df = pd.DataFrame({
    'sentence_1': [pair[0] for pair in sentence_pairs],
    'sentence_2': [pair[1] for pair in sentence_pairs],
    'prediction': predictions
  })

  # is this the same as averaging the three tasks together? (i think so)
  f1 = {"f1": f1_score(labels, predictions, zero_division=0, average='macro')}
  print(f1)

  df.to_csv(f'val-{difficulty}-predictions-1epoch.csv', index=False)
  print("Predictions saved!")

def calc_f1(pred_csv, labels):
  df = pd.read_csv(pred_csv)
  predictions = df['prediction']
  f1 = f1_score(labels, predictions, zero_division=0, average='macro')
  return {"f1": f1}

if __name__ == "__main__":
  args = get_args()

  main(args)