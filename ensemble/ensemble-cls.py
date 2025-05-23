"""

"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

import pickle
import logging
from models import BertPairDataset, BertStyleNN
import argparse
from itertools import chain, combinations

def powerset(iterable):
    # returns set of all subsets of length at least 3 of iterable
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(3, len(s)+1))

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--difficulty", type=str, default="easy")
  parser.add_argument("--ensemble-method", type=str, default="maj-vote", help="One of 'maj-vote', 'avg-logits', 'avg-probs'")
  parser.add_argument("--load-preds", type=bool, default=False)
  parser.add_argument("--ablation", type=bool, default=False)

  return parser.parse_args()

## use to define weights for ensemble
MODEL_STATS = {
    "deberta-base": {"f1": 0.7935, "threshold": 0.68},
    "roberta-base": {"f1": 0.7909, "threshold": 0.89},
    "all-MiniLM-L12-v2": {"f1": 0.7851, "threshold": 0.71},
    # "bert-base-cased": {"f1": 0.7775, "threshold": 0.84},
    "sentence-t5-base": {"f1": 0.7676, "threshold": 0.52},
    "bge-base-en-v1.5": {"f1": 0.7642, "threshold": 0.79},
    "all-mpnet-base-v2": {"f1": 0.7563, "threshold": 0.89}
}

# def run_ensemble(args, models=['deberta-base', 'roberta-base', 'sentence-t5-base', 'all-mpnet-base-v2']):
def run_ensemble(args, models=["all-MiniLM-L12-v2", "deberta-base", "roberta-base", "sentence-t5-base", "bge-base-en-v1.5", "all-mpnet-base-v2"]):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(f'{args.difficulty}_val_pairs.pkl', 'rb') as f:
        val_pairs = pickle.load(f)
    val_labels = np.load(f"{args.difficulty}_val_labels.npy")

    all_outputs = []
    all_preds = []
    labels = None
    logging.info(f"using {args.ensemble_method} for ensemble")
    for path_name in models:
        sentence_transformers = path_name in ["bge-base-en-v1.5", "sentence-t5-base", "all-mpnet-base-v2"]
        logits_loss = args.ensemble_method in ["avg-logits"]
        
        model_name = get_model_name(path_name)
        if not args.ablation:
            logging.info(f"running inference with {model_name}")
        if args.load_preds:
            labels = val_labels
            if not logits_loss:
                output = torch.Tensor(np.load(f"{path_name}_{args.difficulty}_probs.npy"))
            else:
                output = torch.Tensor(np.load(f"{path_name}_{args.difficulty}_logits.npy"))
            _, val_preds = indiv_preds(args, output, labels, logits_loss)
            all_outputs.append(output)
            all_preds.append(val_preds)
            continue

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        val_set = BertPairDataset(tokenizer, val_pairs, val_labels)
        val_set.return_raw_text = sentence_transformers
        val_loader = DataLoader(val_set, batch_size=16, shuffle=False, pin_memory=True)

        # run inference!
        # logits_loss = True means we return logits from forward pass
        criterion = nn.BCEWithLogitsLoss() if logits_loss else nn.BCELoss()

        model = BertStyleNN(enc_model_name=model_name, use_sentence_transformers=sentence_transformers, logits_loss=logits_loss)
        model.load_state_dict(torch.load(f"{path_name}-Best.pth"))
        model.to(device)
        
        loss, output, labels_v = val(model, val_loader, criterion, device)
        if not args.ablation:
            print(f"{model_name} is on {device}")
            logging.info(f"     val loss: {loss}, output {output[0]}")
        
        np.save(f"{path_name}_{args.difficulty}_{args.ensemble_method.split('-')[-1]}.npy", output)
        _, val_preds = indiv_preds(args, output, labels_v, logits_loss)
        
        all_preds.append(val_preds)
        all_outputs.append(output)
        labels = labels_v
    
    if args.ensemble_method == "avg-logits" or args.ensemble_method == "avg-probs":
        metrics, val_preds = ensemble_weighted_outputs(models, all_outputs, labels, apply_sigmoid=logits_loss)
        # metrics, val_preds = ensemble_avg_outputs(all_outputs, labels, apply_sigmoid=logits_loss)
    elif args.ensemble_method == "maj-vote":
        metrics, val_preds = ensemble_majority_voting(all_preds, labels)
    else:
        raise RuntimeError("Ensemble method not supported.")
    
    logging.info(f"{models} ensemble: {metrics}")
    logging.info(f"# of pos preds: {sum(val_preds)}, total: {len(val_preds)}")
    logging.info(f"actual # of pos {sum(val_labels)}")
    return metrics

def indiv_preds(args, outputs, labels, logits_loss):
    # Given model outputs & labels, finds best threshold (by Macro F1) and returns PyTorch tensor of preds and metrics
    # outputs is list for a single model (either logits or probs)
    if logits_loss:
        sigmoid = nn.Sigmoid()
        probs = sigmoid(outputs)
    else:
       probs = outputs

    if torch.is_tensor(labels):
        labels_np = labels.numpy()
    else:
        labels_np = labels

    best_threshold = 0.5  # Default value
    best_macro_f1 = 0
    
    # Test a range of thresholds to find the one that maximizes macro F1
    thresholds = np.linspace(0.05, 0.95, 91)
    for threshold in thresholds:
        preds = (probs >= threshold).int()
        macro_f1 = f1_score(labels_np, preds, average='macro', zero_division=0)
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_threshold = threshold
    
    # use the best threshold to make final predictions
    val_preds = (probs >= best_threshold).int()
    
    # calculate final metrics using the best threshold
    metrics = compute_metrics(y_true=labels_np, y_pred=val_preds, threshold=best_threshold)
    if not args.ablation:
        logging.info(f"     {metrics}")
    return metrics, val_preds

def ensemble_weighted_outputs(models, outputs, labels, apply_sigmoid=False):
    # take weighted avg by each model F1 score during training
    # did not work as well un-weighted average
    f1s = torch.tensor([MODEL_STATS[v]["f1"] for v in models])
    weights = f1s / f1s.sum()
    outputs = torch.stack(outputs)
    weights = weights.view(-1, *([1] * (outputs.ndim - 1)))
    avg_probs = torch.sum(outputs * weights, dim=0)

    # optionally apply sigmoid to turn into probabilities
    if apply_sigmoid:
        sigmoid = nn.Sigmoid()
        avg_probs = sigmoid(avg_probs)
        avg_probs = torch.clamp(avg_probs, min=1e-6, max=1 - 1e-6)
    
    avg_probs_np = avg_probs.numpy()
    if torch.is_tensor(labels):
        labels_np = labels.numpy()
    else:
        labels_np = labels

    # Find threshold that maximizes macro F1
    best_threshold = 0.5  # Default value
    best_macro_f1 = 0
    
    # Test a range of thresholds to find the one that maximizes macro F1
    thresholds = np.linspace(0.05, 0.95, 91)
    for threshold in thresholds:
        preds = (avg_probs_np >= threshold).astype(int)
        macro_f1 = f1_score(labels_np, preds, average='macro', zero_division=0)
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_threshold = threshold
    
    # use the best threshold to make final predictions
    val_preds = (avg_probs_np >= best_threshold).astype(int)
    
    # calculate final metrics using the best threshold
    metrics = compute_metrics(y_true=labels_np, y_pred=val_preds, threshold=best_threshold)
    return metrics, val_preds

def ensemble_majority_voting(all_preds, labels, majority_req=2):
    """
    Given preds (list of lists of binary predictions), implement majority voting where majority_req required to make
    1 prediction (in theory, make it easier to predict 1 due to class imbalance)
    """
    final_preds = (torch.sum(torch.stack(all_preds), dim=0) >= majority_req).int()
    
    # calculate final metrics using the best threshold
    if torch.is_tensor(labels):
        labels = labels.numpy()
    metrics = compute_metrics(y_true=labels, y_pred=final_preds.numpy(), threshold=-1)
    return metrics, final_preds.numpy()

def ensemble_avg_outputs(outputs, labels, apply_sigmoid=False):
    """
    Given list of torch Tensors, calculate avg of outputs (optionally apply sigmoid) and then get predictions from that.
    Finds best threshold for the given validation set (labels).
    """
    # first find avg of the outputs (whether logits or probs)
    avg_probs = torch.mean(torch.stack(outputs), dim=0)

    # optionally apply sigmoid to turn into probabilities
    if apply_sigmoid:
        sigmoid = nn.Sigmoid()
        avg_probs = sigmoid(avg_probs)
        avg_probs = torch.clamp(avg_probs, min=1e-6, max=1 - 1e-6)
    
    avg_probs_np = avg_probs.numpy()
    if torch.is_tensor(labels):
        labels_np = labels.numpy()
    else:
        labels_np = labels

    # Find threshold that maximizes macro F1
    best_threshold = 0.5  # Default value
    best_macro_f1 = 0
    
    # Test a range of thresholds to find the one that maximizes macro F1
    thresholds = np.linspace(0.05, 0.95, 91)
    for threshold in thresholds:
        preds = (avg_probs_np >= threshold).astype(int)
        macro_f1 = f1_score(labels_np, preds, average='macro', zero_division=0)
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_threshold = threshold
    
    # use the best threshold to make final predictions
    val_preds = (avg_probs_np >= best_threshold).astype(int)
    
    # calculate final metrics using the best threshold
    metrics = compute_metrics(y_true=labels_np, y_pred=val_preds, threshold=best_threshold)
    return metrics, val_preds

def ensemble_ablation(args):
    """
    Method which returns subset of models & method with best val performance on the given difficulty partition.
    Tries unweighted avg-probs, unweighted avg-logits, and maj-vote (with default parameters). 
    """
    models = list(MODEL_STATS.keys())
    best_method = "none"
    final_f1 = -1
    final_subset = []
    threshold = float(-1)
    # for method in ["maj-vote", "avg-probs", "avg-logits"]:
    for method in ["avg-probs", "avg-logits"]:
        args.ensemble_method = method
        best_models = []
        best_f1 = -1
        subsets = powerset(models)
        for s in subsets:
            metrics = run_ensemble(args, models=s)
            if metrics['f1'] > best_f1:
                best_models = s
                best_f1 = metrics['f1']
        
        if best_f1 > final_f1:
            best_method = method
            final_f1 = best_f1
            final_subset = best_models
            threshold = metrics['best_threshold']

    ret = {"models": final_subset, "f1": final_f1, "threshold": threshold}    
    logging.info(f"BEST subset for {args.difficulty}, {best_method}")
    logging.info(f"      {ret['models']}, F1: {ret['f1']}, Threshold: {ret['threshold']}")
    return ret

def val(model, val_loader, criterion, device):
    """
    Returns the direct output from the model (whether logits or probs is defined earlier in run_ensemble)
    Outputs are torch tensors on CPU!
    """
    model.eval()
    val_running_loss = 0
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
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
          
          all_outputs.append(outputs.detach().cpu())
          all_labels.append(labels.detach().cpu())
    
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    
    # return logits/probs (as torch.tensors on CPU)
    avg_val_loss = val_running_loss / len(val_loader)
    return avg_val_loss, all_outputs, all_labels

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

def get_model_name(path_name):
    if path_name in ["all-MiniLM-L12-v2", "sentence-t5-base", "all-mpnet-base-v2"]:
        model_name = "sentence-transformers/" + path_name
    elif "deberta" in path_name:
        model_name = "microsoft/" + path_name
    elif "bge" in path_name:
        model_name = "BAAI/" + path_name
    else:
        # roberta-base
        model_name = path_name

    return model_name

if __name__ == '__main__':
    args = get_args()
    if args.ablation:
        logging.basicConfig(
            filename=f'abl_{args.difficulty}.log',
            level=logging.INFO,
            filemode='a', # appends to file
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        ensemble_ablation(args)
    else:  
        logging.basicConfig(
            filename=f'{args.difficulty}_{args.ensemble_method}.log',
            level=logging.INFO,
            filemode='a', # appends to file
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        run_ensemble(args)