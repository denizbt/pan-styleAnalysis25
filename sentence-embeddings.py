import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
# from transformers import Trainer, TrainingArguments

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

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
      print(f"original loaded size {embs.size()}")
      self.embeddings = embs.view(embs.size(0), -1)
      self.labels = torch.tensor(labels, dtype=torch.float32)
      
      print(f"embeddings size {self.embeddings.size()}")
      print(f"labels size {self.labels.size()}")
      raise RuntimeError("stop get some help")

    def __len__(self):
      return self.embeddings.size(0)

    def __getitem__(self, idx):
      embedding = self.embeddings[idx]
      label = self.labels[idx]

      return embedding, label

"""
MLP for binary classification of sentences for same author or not.
"""
class StyleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=150, output_dim=1):
        super(StyleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc3 = nn.Linear(hidden_dim//2, output_dim)
        self.sigmoid = nn.Sigmoid()  # for binary classification
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

"""
Training loop for StyleNN
"""
def train_val_NN(train_data_path, train_labels, val_data_path, val_labels, num_epochs=5, threshold=0.5):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  train_set = SentPairDataset(train_data_path, train_labels)
  val_set = SentPairDataset(val_data_path, val_labels)

  train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
  val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

  model = StyleNN(input_dim=train_set.embeddings.size(1)) ## TODO check if this is the right dim
  model.to(device)
  
  optimizer = optim.AdamW(model.parameters()) # default lr=0.001
  criterion = nn.BCELoss() # or weighted cross entropy?

  best_val_loss = float("inf")
  patience = 3  # stop if no improvement after 3 epochs
  patience_counter = 0
  best_epoch = 0

  for e in tqdm(range(num_epochs), desc="epochs"):
    train_running_loss = 0
    for inputs, labels in tqdm(train_loader, desc=f"train loader {e}", leave=False):
      inputs = inputs.to(device)
      labels = labels.to(device) # BCE requires floats for labels
      print("Batch input shape:", inputs.shape)  # (batch_size, 2*embedding_dim)
      print("Batch labels shape:", labels.shape) # (batch_size,)
      #break
      
      model.train()
      optimizer.zero_grad()
      outputs = model(inputs).squeeze()
      loss = criterion(outputs, labels) 
      loss.backward()
      optimizer.step()

      train_running_loss += loss.item()
    
    model.eval()
    val_running_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
      for inputs, labels in tqdm(val_loader, desc=f"val loader {e}", leave=False):
        inputs, labels = inputs.to(device), labels.to(device).float()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)

        val_running_loss += loss.item()
        preds = (outputs >= threshold).float()
        
        all_preds.extend(preds)
        all_labels.extend(labels)
        
    metrics = compute_metrics(torch.stack(all_preds), torch.stack(all_labels))
    avg_train_loss = train_running_loss / len(train_loader)
    avg_val_loss = val_running_loss / len(val_loader)
    print(f"epoch {e}:\ntraining loss:{avg_train_loss:.4f}\nval loss:{avg_val_loss:.4f}")
    print(f"metrics: {metrics}\n")

    # save model after every epoch
    torch.save(model.state_dict(), f"mlp_epoch_{e}.pth")

    if avg_val_loss < best_val_loss:
      best_val_loss = avg_val_loss
      patience_counter = 0
      best_epoch = e
    else:
      patience_counter += 1
    
    # Early stopping condition: if patience exceeds the limit, stop training
    if patience_counter >= patience:
      print(f"early stopping triggered after {e+1} epochs.")
      break
    
  print(f"training over! best epoch was {best_epoch}")


def compute_metrics(labels, preds):
  labels_np = labels.cpu().numpy()
  preds_np = preds.cpu().numpy()
  
  return {
      'accuracy': accuracy_score(labels_np, preds_np),
      'precision': precision_score(labels_np, preds_np),
      'recall': recall_score(labels_np, preds_np),
      'f1_score': f1_score(labels_np, preds_np)
  }

def save_embeddings(data, split="train"):
  # model_name = "sentence-transformers/all-MiniLM-L12-v2"
  model_name = "princeton-nlp/unsup-simcse-roberta-base"
  file_path = model_name.split("/")[1]
  print(file_path)
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModel.from_pretrained(model_name).eval()

  device = "cuda" if torch.cuda.is_available() else "cpu"
  model.to(device)

  all_embeddings = []
  for s1, s2 in tqdm(data):
    input1 = tokenizer(s1, padding=True, truncation=True, return_tensors="pt").to(device)
    input2 = tokenizer(s2, padding=True, truncation=True, return_tensors="pt").to(device)

    # embeddings.shape: len(d) x embedding dim
    # type(embeddings) = torch.Tensor
    with torch.no_grad():
      emb1 = model(**input1, output_hidden_states=True, return_dict=True).pooler_output
      emb2 = model(**input2, output_hidden_states=True, return_dict=True).pooler_output

    # with torch.no_grad():
    #   out1 = model(**input1)
    #   out2 = model(**input2)

    # # Perform pooling and normalize
    # emb1 = mean_pooling(out1, input1['attention_mask'])
    # emb2 = mean_pooling(out2, input2['attention_mask'])
    # emb1 = F.normalize(emb1, p=2, dim=1)
    # emb2 = F.normalize(emb2, p=2, dim=1)
    # print(f"s1: {emb1.size()}, s2: {emb2.size()}")
    
    # put two pairs together (new dimension, not concatenation)
    pair = torch.stack([emb1, emb2], dim=0)
    # print(f"pair: {pair.size()}")

    # now save the embeddings
    all_embeddings.append(pair.cpu())

  sentence_embeddings = torch.stack(all_embeddings, dim=0).squeeze()
  print(f"sentence_embeddings {sentence_embeddings.size()}")
  print(f"# of sentence pairs {len(data)}")

  # final tensor should have dim : (# of sentences) x 2 x (embedding dim)
  torch.save(sentence_embeddings, f"TEST_{file_path}_{split}.pt")
  # torch.save(sentence_embeddings, f"{file_path}_{split}_embeddings.pt")

def main():
  train_sentences, train_pairs, train_labels = get_data("train")
  val_sentences, val_pairs, val_labels = get_data("validation")
  print("data extracted!")
  print(f"train: # of sentences {len(train_sentences)}, # of pairs {len(train_pairs)}, # of labels {len(train_labels)}")
  print(f"val: # of sentences {len(val_sentences)}, # of pairs {len(val_pairs)}, # of labels {len(val_labels)}")
  # TODO figure out how to save the extracted data so don't need to run this code everytime
  # dataset lengths are good up till here!
  
  save_embeddings(val_pairs, split="val")
  print("saved validation!")

  save_embeddings(train_pairs, split="train")
  print("saved training!")
  
  # train_path = "unsup-simcse-roberta-base_train_embeddings.pt"
  # val_path = "unsup-simcse-roberta-base_val_embeddings.pt"
  # train_val_NN(train_path, train_labels, val_path, val_labels)

if __name__ == "__main__":
  main()