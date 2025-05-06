# doesn't include reading in data
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

import pickle

from models import BertPairDataset, BertStyleNN

def ensemble_preds(difficulty):
 device = "cuda" if torch.cuda.is_available() else "cpu"

 with open(f'{difficulty}_val_pairs.pkl', 'rb') as f:
  val_pairs = pickle.load(f)
 
 val_labels = np.load(f"{difficulty}_val_labels.npy")

 models = ["all-MiniLM-L12-v2", "deberta-base", "roberta-base", "sentence-t5-base", "bge-base-en-v1.5"]
 probabilities = []
 for model_name in models:
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   val_set = BertPairDataset(tokenizer, val_pairs, val_labels)
   sentence_transformers = model_name in ["bge-base-en-v1.5", "sentence-t5-base"]
   val_set.return_raw_text = sentence_transformers

   val_loader = DataLoader(val_set, batch_size=16, shuffle=False, pin_memory=True)

   # run inference!
   model = BertStyleNN(enc_model_name=model_name, use_sentence_transformers=sentence_transformers)
   model.load_state_dict(f"{model_name}-Best.pth")
   model.to(device)
   print(f"model is on {device}")

   metrics, loss, preds = val(model, val_loader, nn.BCEWithLogitsLoss(), device)
   print(f"run inference with {model_name}, \n{metrics},\n val loss: {loss}")
   # np.save(f"{model_name}_preds_ensemble.npy", preds)
    

def val(model, val_loader, criterion, device):
   # TODO write this without thresholding
   # return probabilities
   pass

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

if __name__ == '__main__':
    ensemble_preds()