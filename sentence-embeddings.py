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

# for mini-lm
#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# function which calculates cosine similarity between sentence embeddings
def cosine_sim_classification(documents, labels):
  model_name = "sentence-transformers/all-MiniLM-L12-v2"
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
  
  print(f"{model_name} performance")
  for i in range(len(thresholds)):
    macro_f1 = f1_score(labels, predictions[i], average='macro')
    print(f"{thresholds[i]} Macro F1: {macro_f1}")


"""
Custom Dataset for sentence pair classification
"""
class SentPairDataset(Dataset):
    def __init__(self, embeddings_path, labels):
      # TODO try training with 2x training data i.e. concatenation both ways.
      embs = torch.load(embeddings_path)
      self.embeddings = embs.view(embs.size[0] - 1, -1)
      self.labels = labels

    def __len__(self):
      return self.embeddings.size[0]

    def __getitem__(self, idx):
      embedding = self.embeddings[idx]
      label = self.labels[idx]

      return embedding, label

"""
MLP for binary classification of sentences for same author or not.
"""
class StyleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(StyleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim/2)
        self.fc3 = nn.Linear(hidden_dim/2, 2)
        self.relu = nn.ReLU() ## TODO decide whether ouput dim should be 1 or 2
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # if using Cross Entropy Loss, do not need to apply softmax. otherwise yes!
        return x

"""
Training loop for StyleNN
"""
def train_val_NN(train_data_path, train_labels, val_data_path, val_labels, num_epochs=5):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  train_set = SentPairDataset(train_data_path, train_labels)
  val_set = SentPairDataset(val_data_path, val_labels)

  train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
  val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

  model = StyleNN()
  model.to(device)
  
  optimizer = optim.Adam(model.parameters()) # default lr=0.001
  criterion = nn.BCELoss()

  train_loss = []
  best_val_loss = float("inf")
  patience = 3  # stop if no improvement after 3 epochs
  counter = 0

  # TODO finish training loop
  for epoch in len(num_epochs):
    for inputs, labels in train_loader:
      inputs = inputs.to(device)
      labels = labels.to(device).float() # BCE requires floats for labels

      print("Batch input shape:", inputs.shape)  # (batch_size, 2*embedding_dim)
      print("Batch labels shape:", labels.shape) # (batch_size,)
      #break

      model.train()
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels) 
      loss.backward()
      optimizer.step()

      train_loss.append(loss.item())
    
    # TODO at end of epoch, calc. val loss
    model.eval()
    num_correct = 0
    val_running_loss = 0
    with torch.no_grad():
      for inputs, labels in val_loader:
          inputs, labels = inputs.to(device), labels.to(device).float()
          outputs = model(inputs)
          loss = criterion(outputs, labels)

          val_running_loss += loss.item()

          # TODO this assumes that final dim of output is > 1, and picks index with higher probability
          # decide whether or not we are using higher dim or not
          _, predictions = torch.max(outputs, 1)
          correct = (predictions == labels)
          num_correct += correct.sum()


def save_embeddings(data, split="train"):
  model_name = "princeton-nlp/unsup-simcse-roberta-large"
  file_path = model_name.split("/")[1]
  print(file_path)
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModel.from_pretrained(model_name).eval()

  device = "cuda" if torch.cuda.is_available() else "cpu"
  model.to(device)

  all_embeddings = []
  for d in tqdm(data):
    inputs = tokenizer(d, padding=True, truncation=True, return_tensors="pt").to(device)

    # embeddings.shape: len(d) x embedding dim
    # type(embeddings) = torch.Tensor
    with torch.no_grad():
      embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

    # with torch.no_grad():
    #   model_output = model(**inputs)

    # Perform pooling and normalize
    #embeddings = mean_pooling(model_output, inputs['attention_mask'])
    #embeddings = F.normalize(embeddings, p=2, dim=1)
     
    # embeddings is list of sentence embeddings, now create pairs
    pairs = torch.stack([torch.stack((embeddings[i], embeddings[i+1])) for i in range(len(embeddings) - 1)])

    # now save the embeddings
    all_embeddings.append(pairs.cpu())

  sentence_embeddings = torch.cat(all_embeddings, dim=0)

  # final tensor should have dim : (# of sentences) x 2 x (embedding dim) [or smth like that]
  torch.save(sentence_embeddings, f"{file_path}_{split}_embeddings.pt")

def main():
  train_sentences, train_pairs, train_labels = get_data("train")
  val_sentences, val_pairs, val_labels = get_data("validation")
  print("data extracted!")

  save_embeddings(train_sentences, split="train")
  print("saved training!")

  save_embeddings(val_sentences, split="val")
  print("saved validation!")
  # need to run this for val and training!

if __name__ == "__main__":
  main()