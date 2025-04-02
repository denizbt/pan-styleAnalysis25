import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, precision_recall_curve

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
        embs = torch.load(embeddings_path)
        self.embeddings = embs.view(embs.size(0), -1)
        
        labels = labels.tolist() if type(labels) is not list else labels
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return self.embeddings.size(0)

    def __getitem__(self, idx):
      return self.embeddings[idx], self.labels[idx]

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
Training loop for StyleNN, also calls val() loop
"""
def train(train_data_path, train_labels, val_data_path, val_labels, batch_size=64, num_epochs=1, threshold=0.5):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  train_set = SentPairDataset(train_data_path, train_labels)
  val_set = SentPairDataset(val_data_path, val_labels)

  train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
  val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

  embedding_dim = train_set.embeddings.size(1)
  model = StyleNN(input_dim=embedding_dim)
  model.to(device)
  
  optimizer = optim.AdamW(model.parameters()) # default lr=0.001
  criterion = nn.BCELoss()

  best_val_f1 = -1
  patience = 3  # stop if no improvement after some # of epochs
  patience_counter = 0
  best_epoch = 0

  for e in tqdm(range(num_epochs), desc="Epochs", position=0):
    train_running_loss = 0
    for inputs, labels in tqdm(train_loader, desc=f"train batches (epoch {e+1})", position=1, leave=False):
      inputs = inputs.to(device)
      labels = labels.to(device)
      
      model.train()
      optimizer.zero_grad()
      outputs = model(inputs).view(-1)
  
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      train_running_loss += loss.item()

    # save model after every training epoch
    torch.save(model.state_dict(), f"mlp_epoch_{e}.pth")  

    avg_train_loss = train_running_loss / len(train_loader)
    metrics, avg_val_loss = val(model, val_loader, criterion, device)
    print(f"\nepoch {e}\ntraining loss: {avg_train_loss:.4f}\nval loss: {avg_val_loss:.4f}")
    print(f"val metrics: {metrics}\n")

    if metrics['f1'] < best_val_f1:
        best_val_f1 = metrics['f1']
        patience_counter = 0
        best_epoch = e
    else:
      patience_counter += 1
      
    # Early stopping condition: if patience exceeds the limit, stop training
    if patience_counter >= patience:
      print(f"early stopping triggered after {e+1} epochs.")
      break
    
  print(f"training over! best epoch was {best_epoch}")


def val(model, val_loader, criterion, device):
    model.eval()
    val_running_loss = 0
    all_outputs = []
    all_labels = []
    with torch.no_grad():
      for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device).float()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)

        val_running_loss += loss.item()
        
        all_outputs.extend(outputs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)
    precision, recall, thresholds = precision_recall_curve(all_labels, all_outputs)
        
    # Calculate F1 scores
    # Note: precision_recall_curve returns one more precision/recall value than thresholds
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    
    # Find threshold with best F1 score
    if len(f1_scores) > 0:  # Make sure we have valid F1 scores
        best_idx = np.argmax(f1_scores)
        epoch_threshold = thresholds[best_idx]
    else:
        epoch_threshold = 0.5
    
    # apply the best f1 threshold to make predictions
    val_predictions = (all_outputs >= epoch_threshold).astype(int)
    
    metrics = compute_metrics(y_true=all_labels, y_pred=val_predictions, threshold=epoch_threshold)
    avg_val_loss = val_running_loss / len(val_loader)
    return metrics, avg_val_loss


def compute_metrics(y_true, y_pred, threshold):
    """
    Compute classification metrics using a given threshold
    
    Args:
        y_true: Ground truth labels
        y_pred: Predictions from model (0, 1)
        threshold: Threshold to use for binary classification
        
    Returns:
        Dictionary with accuracy, precision, recall, f1 score, and threshold applied
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'best_threshold': threshold
    }
    
    return metrics

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
  # print(f"sentence_embeddings {sentence_embeddings.size()}")
  # print(f"# of sentence pairs {len(data)}")

  # final tensor should have dim : (# of sentences) x 2 x (embedding dim)
  torch.save(sentence_embeddings, f"{file_path}_{split}.pt")

def main():
  # train_sentences, train_pairs, train_labels = get_data("train")
  # val_sentences, val_pairs, val_labels = get_data("validation")
  # print("data extracted!")
  # print(f"train: # of sentences {len(train_sentences)}, # of pairs {len(train_pairs)}, # of labels {len(train_labels)}")
  # print(f"val: # of sentences {len(val_sentences)}, # of pairs {len(val_pairs)}, # of labels {len(val_labels)}")

  # save_embeddings(val_pairs, split="val")
  # print("saved validation!")

  # save_embeddings(train_pairs, split="train")
  # print("saved training!")
  
  train_labels = np.load("train_labels.npy")
  val_labels = np.load("val_labels.npy")
  train_path = "all-MiniLM-L12-v2_train.pt"
  val_path = "all-MiniLM-L12-v2_val.pt"
  train(train_path, train_labels, val_path, val_labels)

if __name__ == "__main__":
  main()