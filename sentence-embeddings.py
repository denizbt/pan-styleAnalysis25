import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score

from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

from tqdm import tqdm

# utils.py
from utils import read_labeled_data, pair_sentences_with_labels

# for mini-lm
#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def cosine_sim_classification(documents, labels, threshold=0.5):
  model_name = "sentence-transformers/all-MiniLM-L6-v2"
  # tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
  # model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModel.from_pretrained(model_name)

  # iterate through the sentences and compare cosine sims of embeddings
  thresholds = [0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98]
  predictions = [[] for _ in range(len(thresholds))] # list of predictions
  
  for d in tqdm(documents):
    preds = [[] for _ in range(len(thresholds))]
    inputs = tokenizer(d, padding=True, truncation=True, return_tensors="pt")

    # embeddings.shape: len(d) x embedding dim
    # type(embeddings) = torch.Tensor
    # with torch.no_grad():
    #   embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

    # Compute token embeddings
    with torch.no_grad():
      model_output = model(**inputs)

    # Perform pooling
    embeddings = mean_pooling(model_output, inputs['attention_mask'])

    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    assert embeddings.shape[0] == len(d)
    for i in range(len(d)-1):
      cosine_sim = cosine(embeddings[i], embeddings[i+1])

      for i in range(len(thresholds)):
        if cosine_sim > thresholds[i]:
          # same author
          preds[i].append(1)
        else:
          preds[i].append(0)
    
    # assert len(preds) == (len(d)-1)
    for i in range(len(thresholds)): 
      predictions[i].extend(preds[i])
  
  for i in range(len(thresholds)):
    macro_f1 = f1_score(labels, predictions[i], average='macro')
    print(f"{thresholds[i]} Macro F1: {macro_f1}")


# pytorch NN for classification
class StyleNN(nn.Module):
    def __init__(self, input_dim):
        super(StyleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # binary classification
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))  # output probability
        return x

def get_data(dataset_split):
     # Load data (Assuming sentence_pairs contains embeddings & labels contains 0 or 1)
    easy_probs, easy_labels = read_labeled_data(f"data/easy/{dataset_split}")
    med_probs, med_labels = read_labeled_data(f"data/medium/{dataset_split}")
    hard_probs, hard_labels = read_labeled_data(f"data/hard/{dataset_split}")

    # now create pairs of sentences, in pandas dictionary
    easy_pairs, easy_labels = pair_sentences_with_labels(easy_probs, easy_labels)
    med_pairs, med_labels = pair_sentences_with_labels(med_probs, med_labels)
    hard_pairs, hard_labels = pair_sentences_with_labels(hard_probs, hard_labels)

    all_sentences = easy_probs + med_probs + hard_probs
    all_pairs = easy_pairs + med_pairs + hard_pairs
    all_labels = easy_labels + med_labels + hard_labels

    return all_sentences, all_pairs, all_labels

def main():
  train_sentences, train_pairs, train_labels = get_data("train")
  val_sentences, val_pairs, val_labels = get_data("validation")
  print("data extracted!")

  cosine_sim_classification(val_sentences, val_labels)

if __name__ == "__main__":
  main()