import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, precision_recall_curve

from tqdm import tqdm
import json
import argparse

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--embedding-type", type=str, default="all-MiniLM-L12-v2"
  )
  
  return parser.parse_args()


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
def train(args, train_data_path, train_labels, val_data_path, val_labels, batch_size=64, num_epochs=15, patience=3):
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

  best_metrics = {'f1': -1}
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
    torch.save(model.state_dict(), f"{args.embedding_type}_mlp_epoch_{e}.pth")  

    avg_train_loss = train_running_loss / len(train_loader)
    metrics, avg_val_loss = val(model, val_loader, criterion, device)
    print(f"\nepoch {e}\ntraining loss: {avg_train_loss:.4f}\nval loss: {avg_val_loss:.4f}")
    print(f"val metrics: {metrics}\n")

    if metrics['f1'] > best_metrics['f1']:
        best_metrics = metrics
        patience_counter = 0
        best_epoch = e
    else:
      patience_counter += 1
      
    # Early stopping condition: if patience exceeds the limit, stop training
    if patience_counter >= patience:
      print(f"early stopping triggered after {e+1} epochs.")
      break
    
  with open(f"{args.embedding_type}_val_metrics.json", "w+") as f:
     best_metrics = {k: float(v) if hasattr(v, 'item') else v for k, v in best_metrics.items()}
     json.dump(best_metrics, f)
  
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
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "best_threshold": threshold
    }
    
    return metrics

def main(args):
  # load saved labels (numpy array of changes, 0 or 1)
  train_labels = np.load("train_labels.npy")
  val_labels = np.load("val_labels.npy")
  
  # load saved sentence embeddings
  train_path = args.embedding_type+"_train.pt"
  val_path = args.embedding_type+"_val.pt"
  train(args, train_path, train_labels, val_path, val_labels)

if __name__ == "__main__":
  args = get_args()
  main(args)