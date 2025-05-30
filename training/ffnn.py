"""
Script which defines a MLP/FFNN used as binary sequence classification head for BertStyleNN (training/bert-training.py).
  - Includes train and validation functions to train just FFBB, assuming static sentence embeddings extracted (as torch tensors)
  - Architecture chosen through validation testing with sentence-transformers/all-MiniLM-L12-v2 using PAN 2025 Multi-Author cls data.
"""

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from tqdm import tqdm
import json
import argparse

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--embedding-type", type=str, default="all-MiniLM-L12-v2")
  parser.add_argument("--weighted-loss", type=bool, default=False)  
  parser.add_argument("--reverse-augment", type=bool, default=False)
  parser.add_argument("--lr-schedule", type=bool, default=False)
  parser.add_argument("--balanced-train", type=bool, default=False)

  return parser.parse_args()


class SentPairDataset(Dataset):
    """
    Custom (Pytorch) Dataset for sentence pair classification
    """
    def __init__(self, embeddings_path, labels, direct_pass=False, reverse_augment=False):
      """
      embeddings_path: path to .pt file, or torch.Tensor
      direct_pass: True when embeddings_path contains torch.Tensor
      reverse_augment: True when we want to add reverse concatenation of embeddings to the dataset
      """
      if direct_pass:
        embs = embeddings_path
      else:
        embs = torch.load(embeddings_path)
      
      # embs shape: (total_pairs, 2, embedding_dim)
      if reverse_augment:
        forward_concat = torch.cat([embs[:, 0, :], embs[:, 1, :]], dim=1)            
        reverse_concat = torch.cat([embs[:, 1, :], embs[:, 0, :]], dim=1)
        
        # add both directations
        self.embeddings = torch.cat([forward_concat, reverse_concat], dim=0)
        
        # duplicate labels for the reversed concatentation
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        self.labels = torch.cat([labels_tensor, labels_tensor], dim=0)
      else:
        self.embeddings = embs.view(embs.size(0), -1)
        self.labels = torch.tensor(labels, dtype=torch.float32)      

    def __len__(self):
      return self.embeddings.size(0)

    def __getitem__(self, idx):
      return self.embeddings[idx], self.labels[idx]


class StyleNN(nn.Module):
    """
    Feed-forward neural network (FFNN) for binary classification of sentences for same author or not.
    """
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64], output_dim=1, p=0.4, apply_sigmoid=True):
        """
        Args:
          p: dropout rate
          apply_sigmoid: True if model should apply sigmoid as last step in forward()
        """
        super(StyleNN, self).__init__()
        layers = []
        
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.Dropout(p))
        
        # define all hidden layers
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            layers.append(nn.Dropout(p))
        
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.model = nn.Sequential(*layers)
        
        self.sigmoid = nn.Sigmoid()
        self.apply_sigmoid = apply_sigmoid
        
    def forward(self, x):
      x = self.model(x)
      
      # apply sigmoid if not using BCEWithLogitsLoss
      if self.apply_sigmoid:
        x = self.sigmoid(x)
      
      return x

def train_mlp(args, train_data_path, train_labels, val_data_path, val_labels, batch_size=64, num_epochs=15, patience=3):
  """
  Training loop for StyleNN, also calls val() loop
  """
  device = "cuda" if torch.cuda.is_available() else "cpu"
  train_set = SentPairDataset(train_data_path, train_labels, reverse_augment=args.reverse_augment)
  val_set = SentPairDataset(val_data_path, val_labels)

  train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
  val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

  embedding_dim = train_set.embeddings.size(1)
  model = StyleNN(input_dim=embedding_dim, apply_sigmoid=not(args.weighted_loss))
  model.to(device)
  
  lr = 0.001  # default = 0.01
  optimizer = optim.AdamW(model.parameters(), lr=lr)
  if args.lr_schedule:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
  
  if args.weighted_loss:
    # calculate positive class weight
    # training, and val thresholds get super unstable when i use weights
    num_class_1 = sum(train_set.labels)
    pos_weight = (len(train_set.labels)-num_class_1) / num_class_1
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
  else:
    criterion = nn.BCELoss()

  best_val_preds = None
  best_metrics = {'f1': -1}
  patience_counter = 0
  best_epoch = 0
  best_model_state = None

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
      
      # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # grad clipping
      optimizer.step()

      train_running_loss += loss.item()
    
    avg_train_loss = train_running_loss / len(train_loader)
    metrics, avg_val_loss, val_preds = val_mlp(model, val_loader, criterion, device)
    print(f"\nepoch {e}\ntraining loss: {avg_train_loss:.4f}\nval loss: {avg_val_loss:.4f}")
    print(f"val metrics: {metrics}\n")
    
    # update learning rate using scheduler
    if args.lr_schedule:
      scheduler.step(avg_val_loss)

    if metrics['f1'] > best_metrics['f1']:
        best_val_preds = val_preds
        best_metrics = metrics
        patience_counter = 0
        best_epoch = e
        best_model_state = model.state_dict().copy()
    else:
      patience_counter += 1
      
    # early stopping condition: if patience exceeds the limit, stop training
    if patience_counter >= patience:
      print(f"early stopping triggered after {e+1} epochs.")
      break

  # save metrics, preds, model
  file_name = f"{args.embedding_type}_"
  if args.balanced_train:
    file_name += "bal_"
  if args.weighted_loss:
    file_name += "weighted_"
  if args.reverse_augment:
    file_name += "reverse_"
  if args.lr_schedule:
    file_name += "lr_"

  with open(f"{file_name}metrics.json", "w+") as f:
     best_metrics = {k: float(v) if hasattr(v, 'item') else v for k, v in best_metrics.items()}
     json.dump(best_metrics, f)
  
  np.save(f"{file_name}preds.npy", best_val_preds)

  model.load_state_dict(best_model_state)
  torch.save(model.state_dict(), f"{file_name}mlp_model.pth")
  print(f"training over! best epoch was {best_epoch}")

def val_mlp(model, val_loader, criterion, device):
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
    
    # Find threshold that maximizes macro F1
    best_threshold = 0.5  # Default value
    best_macro_f1 = 0
    
    # Test a range of thresholds to find the one that maximizes macro F1
    thresholds = np.linspace(0.1, 0.9, 81)
    
    for threshold in thresholds:
        preds = (all_outputs >= threshold).astype(int)
        macro_f1 = f1_score(all_labels, preds, average='macro', zero_division=0)
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_threshold = threshold
    
    # use the best threshold to make final predictions
    val_preds = (all_outputs >= best_threshold).astype(int)
    
    # calculate final metrics using the best threshold
    metrics = compute_metrics(y_true=all_labels, y_pred=val_preds, threshold=best_threshold)
    
    avg_val_loss = val_running_loss / len(val_loader)
    return metrics, avg_val_loss, val_preds

def compute_metrics(y_true, y_pred, threshold):
    """
    Compute classification metrics (accuracy, precision, recall, f1) using the given threshold
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
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
  # load saved labels (numpy array of changes)
  # 0 if there was no style change, 1 if there was a style change
  if args.balanced_train:
    train_path = args.embedding_type+"_bal_train.pt"
    train_labels = np.load("train_bal_labels.npy")
  else:  
    train_path = args.embedding_type+"_train.pt"
    train_labels = np.load("train_labels.npy")
  
  val_labels = np.load("val_labels.npy")
  
  # load saved sentence embeddings
  val_path = args.embedding_type+"_val.pt"
  train_mlp(args, train_path, train_labels, val_path, val_labels)

if __name__ == "__main__":
  args = get_args()
  main(args)