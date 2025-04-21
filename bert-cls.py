import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from tqdm import tqdm
import json
import numpy as np
import pickle
import logging

# mlp.py, file i wrote
from mlp import StyleNN

import argparse

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-name", type=str, default="roberta-base")
  parser.add_argument("--data-dir", type=str, default="data/")
  parser.add_argument("--workers", type=int, default=1)
  parser.add_argument("--pooling", type=str, default="mean")
  parser.add_argument("--resume-training", type=str, default="None")
  parser.add_argument("--run-inference", type=str, default="")
  parser.add_argument("--sentence-transformers", type=bool, default=False)
  
  return parser.parse_args()

class BertStyleNN(nn.Module):
  """
  NN which uses BERT(etc.) encoder and separate MLP for classification.
  1. Uses self.encoder for independent feature extraction of two sentences.
  2. Concatenates the two embeddings, and passes through StyleNN (defined in mlp.py) for final classification.
  """
  def __init__(self, hidden_dims=[512, 256, 128, 64], output_dim=1, enc_model_name='roberta-base', pooling='mean', use_sentence_transformers=False):
    """
    Args:
      hidden_dims [List[int]]:
      output_dim [int]: final classification of the model is a single dimension
      pooling [str] in ['mean', 'cls]
      resume_training: If not None, contains path to encoder-only model state dict from which to resume training
    """
    super(BertStyleNN, self).__init__()  

    # check if it's a sentence transformers model
    # self.sentence_transformers = use_sentence_transformers
    if use_sentence_transformers:
      self.encoder = SentenceTransformer(enc_model_name)
      # self.encoder = SentenceTransformer(enc_model_name, trust_remote_code=True, model_kwargs={'default_task': 'text-matching'})
      with torch.no_grad():
        dummy_embedding = self.encoder.encode(["Hello"], convert_to_tensor=True)
        embedding_dim = dummy_embedding.shape[-1]
    else:
      self.encoder = AutoModel.from_pretrained(enc_model_name)  
      embedding_dim =  self.encoder.config.hidden_size

    self.pooling = pooling
    # input to MLP is the concatenation of the sentence pairs extracted
    self.mlp = StyleNN(input_dim=embedding_dim*2, hidden_dims=hidden_dims, output_dim=output_dim)
  
  def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
    # extract features from encoder independently on the two sentences
    if attention_mask1 is None:
      s1 = self.encoder.encode(input_ids1, convert_to_tensor=True, batch_size=16, show_progress_bar=False).to(next(self.parameters()).device)
      s2 = self.encoder.encode(input_ids2, convert_to_tensor=True, batch_size=16, show_progress_bar=False).to(next(self.parameters()).device)
    else:
      s1 = self.encoder(input_ids=input_ids1, attention_mask=attention_mask1)
      s2 = self.encoder(input_ids=input_ids2, attention_mask=attention_mask2)
      
      if self.pooling == 'mean':
        s1 = mean_pooling(s1, attention_mask1)
        s2 = mean_pooling(s2, attention_mask2)
      else:
        # pooler_output takes [CLS] hidden layer vector
        s1 = s1.pooler_output if s1.pooler_output is not None else s1.last_hidden_state[:, 0, :]
        s2 = s2.pooler_output if s2.pooler_output is not None else s2.last_hidden_state[:, 0, :]
    
    # concatenate features from sentece pairs to pass into MLP for classification
    # using BCEWithLogitsLoss, so no sigmoid
    concat = torch.cat((s1, s2), dim=1)
    logits = self.mlp(concat)
    return logits

def mean_pooling(model_output, attention_mask):
  token_embeddings = model_output.last_hidden_state  # [batch_size, seq_len, hidden_dim]
  input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
  sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
  sum_mask = input_mask_expanded.sum(1)
  return sum_embeddings / sum_mask.clamp(min=1e-9)  # prevent division by zero

class BertPairDataset(Dataset):
  """
  Custom (PyTorch) dataset
  """
  def __init__(self, tokenizer, sent_pairs, labels, max_length=175):
    """
    Args:
      tokenizer: the BERT(+) tokenizer to use for the sentence pairs
      sent_pairs List[Tuple]: each element of list is tuple (sentence1, sentence2)
      labels List[int]: 0 or 1, where a 1 indicates style difference between pairs of sentences 
      len(sent_pairs) == len(labels)
    """
    self.tokenizer = tokenizer
    self.sent_pairs = sent_pairs
    self.labels = labels
    self.max_length = max_length
  
  def __len__(self):
    return len(self.labels)
  
  def __getitem__(self, idx):
    s1, s2 = self.sent_pairs[idx]
    label = self.labels[idx]
    
    # return non-tokenized text if sentence-transformers being used
    if self.return_raw_text:
      return {
          's1': s1,
          's2': s2,
          'labels': torch.tensor(label, dtype=torch.float)
      }
    
    # get tokenized values for both sentences in pair
    tok1 = self.tokenizer(
      s1,
      add_special_tokens=True,
      max_length=self.max_length,
      padding='max_length',
      truncation=True,
      return_tensors='pt'
    )
        
    tok2 = self.tokenizer(
      s2,
      add_special_tokens=True,
      max_length=self.max_length,
      padding='max_length',
      truncation=True,
      return_tensors='pt'
    )
    
    return {'input_ids1': tok1['input_ids'].squeeze(0),
          'attention_mask1': tok1['attention_mask'].squeeze(0),
          'input_ids2': tok2['input_ids'].squeeze(0),
          'attention_mask2': tok2['attention_mask'].squeeze(0),
          'labels': torch.tensor(label, dtype=torch.float)}

def train(args, train_pairs, train_labels, val_pairs, val_labels, batch_size=16, num_epochs=15, patience=5):
  """
  Training loop for BertStyleNN
  """
  device = "cuda" if torch.cuda.is_available() else "cpu"
  
  tokenizer = AutoTokenizer.from_pretrained(args.model_name)
  train_set = BertPairDataset(tokenizer, train_pairs, train_labels)
  val_set = BertPairDataset(tokenizer, val_pairs, val_labels)
  train_set.return_raw_text = args.sentence_transformers
  val_set.return_raw_text = args.sentence_transformers
  
  val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
  if args.run_inference != "":
    model = BertStyleNN(enc_model_name=args.model_name, use_sentence_transformers=args.sentence_transformers, pooling=args.pooling)
    model.load_state_dict(torch.load(args.run_inference))
    model.to(device)
    print(f"model is on {device}")
    metrics, loss, preds = val(model, val_loader, nn.BCEWithLogitsLoss(), device)
    logging.info(f"run inference with {args.run_inference}, \n{metrics},\n val loss: {loss}")
    np.save("preds_inference.npy", preds)
    return

  train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
  
  # resume training both encoder and FFNN/MLP weights
  model = BertStyleNN(enc_model_name=args.model_name, use_sentence_transformers=args.sentence_transformers, pooling=args.pooling)
  if args.resume_training != "None":
    logging.info(f"Resuming training from {args.resume_training}") 
    model.load_state_dict(torch.load(args.resume_training))
  
  model.to(device)
  print(f"model is on {device}")
  
  bert_params = list(model.encoder.parameters())
  classifier_params = list(model.mlp.parameters())
  optimizer = torch.optim.AdamW([
      {'params': bert_params, 'lr': 1e-5},  # Lower learning rate for BERT
      {'params': classifier_params, 'lr': 1e-4}  # Higher learning rate for MLP
  ])

  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
  
  pos_weight = torch.tensor([0.8 / 0.2])  # ratio of negative to positive
  pos_weight = pos_weight.to(device)  # move to GPU if using cuda
  criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
  
  best_val_preds = None
  best_metrics = {'f1': -1}
  patience_counter = 0
  best_epoch = 0

  logging.info("starting training!")
  for e in tqdm(range(num_epochs), desc="Epochs", position=0):
    train_running_loss = 0
    for batch in tqdm(train_loader, desc=f"train batches (epoch {e+1})", position=1, leave=False):
      model.train()
      optimizer.zero_grad()
      
      if args.sentence_transformers:  # SentenceTransformer
        input_ids1 = batch['s1']
        input_ids2 = batch['s2']
        outputs = model(input_ids1, None, input_ids2, None).squeeze(1)
      else:  # normal HuggingFace
        input_ids1 = batch['input_ids1'].to(device)
        attention_mask1 = batch['attention_mask1'].to(device)
        input_ids2 = batch['input_ids2'].to(device)
        attention_mask2 = batch['attention_mask2'].to(device)
        outputs = model(input_ids1, attention_mask1, input_ids2, attention_mask2).squeeze(1)

      labels = batch['labels'].to(device)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      train_running_loss += loss.item()
    
    # save model (encoder and entire model) after every epoch
    file_name = args.model_name.split("/")
    file_name = file_name[len(file_name)-1]
    torch.save(model.state_dict(), f"{file_name}-e{e}.pth")
    torch.save(model.encoder.state_dict(), f"enc-only-{file_name}-e{e}.pth")
    
    avg_train_loss = train_running_loss / len(train_loader)
    metrics, avg_val_loss, val_preds = val(model, val_loader, criterion, device)
    
    print(f"\nepoch {e}\ntraining loss: {avg_train_loss:.4f}\nval loss: {avg_val_loss:.4f}")
    logging.info(f"\nepoch {e}\ntraining loss: {avg_train_loss:.4f}\nval loss: {avg_val_loss:.4f}")
    logging.info(f"val metrics: {metrics}\n")
    
    # update learning rate using scheduler
    scheduler.step(avg_val_loss)

    if metrics['f1'] > best_metrics['f1']:
        best_val_preds = val_preds
        best_metrics = metrics
        patience_counter = 0
        best_epoch = e
    else:
      patience_counter += 1
      
    # early stopping condition: if patience exceeds the limit, stop training
    if patience_counter >= patience:
      logging.info(f"early stopping triggered after {e+1} epochs.")
      break
  
  file_name += "_"
  with open(f"{file_name}metrics.json", "w+") as f:
    best_metrics = {k: float(v) if hasattr(v, 'item') else v for k, v in best_metrics.items()}
    json.dump(best_metrics, f)

  np.save(f"{file_name}preds.npy", best_val_preds)

  logging.info(f"training over! best epoch was {best_epoch}")

def val(model, val_loader, criterion, device):
    model.eval()
    val_running_loss = 0
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"val batches", position=1, leave=False):
          if "s1" in batch:  # SentenceTransformer
            input_ids1 = batch["s1"]
            input_ids2 = batch["s2"]
            outputs = model(input_ids1, None, input_ids2, None).squeeze(1)
          else:  # normal HuggingFace
            input_ids1 = batch['input_ids1'].to(device)
            attention_mask1 = batch['attention_mask1'].to(device)
            input_ids2 = batch['input_ids2'].to(device)
            attention_mask2 = batch['attention_mask2'].to(device)
            outputs = model(input_ids1, attention_mask1, input_ids2, attention_mask2).squeeze(1)
            
          labels = batch['labels'].to(device)
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

if __name__ == "__main__":
  args = get_args()
  with open('train_pairs.pkl', 'rb') as f:
    train_pairs = pickle.load(f)
    
  with open('val_pairs.pkl', 'rb') as f:
    val_pairs = pickle.load(f)

  train_labels = np.load("train_labels.npy")
  val_labels = np.load("val_labels.npy")
  # assert len(train_pairs) == len(train_labels)
  # assert len(val_pairs) == len(val_labels)
  
  file_name = args.model_name.split("/")
  file_name = file_name[len(file_name)-1]
  logging.basicConfig(
    filename=f'{file_name}_train.log',
    level=logging.INFO,
    filemode='a', # appends to file
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
  )
  logging.info("read in data.")
  
  torch.cuda.empty_cache() # to reduce memory problems
  train(args, train_pairs, train_labels, val_pairs, val_labels)