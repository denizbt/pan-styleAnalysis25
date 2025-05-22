#!/usr/bin/env python3
import json
import torch
import logging
from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import click

from models import BertPairDataset, BertStyleNN
import torch.nn as nn
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# global variables (found from validation set experimentation)
BEST_THRESHOLDS = {
    "deberta-base": 0.68,
    "roberta-base": 0.89,
    "all-MiniLM-L12-v2": 0.71,
    "sentence-t5-base": 0.52,
    # "bge-base-en-v1.5": 0.79,
    "all-mpnet-base-v2": 0.89
}

ENSEMBLE_THRESHOLDS = {"easy": 0.6699999999999999, "medium": 0.58, "hard": 0.57}
ENSEMBLE_MODELS = {"easy": ['deberta-base', 'all-MiniLM-L12-v2', 'sentence-t5-base'],
                   "medium": ['deberta-base', 'roberta-base', 'all-mpnet-base-v2'],
                   "hard": ['deberta-base', 'roberta-base', 'all-MiniLM-L12-v2', 'all-mpnet-base-v2']}
ENSEMBLE_METHODS = {"easy": "avg-logits", "medium": "avg-probs", "hard": "avg-probs"}

def run_problems(difficulty, problems, output_path):
    """
    Process problems and write predictions using your model.
    
    Args:
        difficulty (str): "easy", "medium", or "hard"
        problems: DataFrame of problem files
        output_path: Path to write solution files
    """
    logger.info(f'Processing {len(problems)} problems and writing outputs to {output_path}.')
    
    for _, item in problems.iterrows():
        # create output file path
        output_file = output_path / item["file"].replace("/problem-", "/solution-problem-").replace(".txt", ".json").replace("/train/", "/").replace("/test/", "/").replace("/validation/", "/")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # paragraphs is the list of sentences for a specific file
        paragraphs = item["paragraphs"]
        predictions = run_ensemble(difficulty, paragraphs)
        
        # write predictions to output file
        with open(output_file, 'w') as out:
            prediction = {'changes': predictions}
            out.write(json.dumps(prediction))
            
        logger.info(f"Processed {Path(item['file']).name} successfully")
        
        # running many models, so clear cache frequently
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def run_ensemble(difficulty, paragraphs):
    """
    Runs all of the models, uses avg logits with best threshold to return final binary predictions.

    Args:
        difficulty (str): one of "easy", "medium", "hard"
        paragraphs (list??): test set input

    Returns:
        preds: list of final binary predictions for this test set
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models = ENSEMBLE_MODELS[difficulty]
    all_outputs = []
    all_preds = []
    for path_name in models:
        sentence_transformers = path_name in ["bge-base-en-v1.5", "sentence-t5-base", "all-mpnet-base-v2"]
        logits_loss = ENSEMBLE_METHODS[difficulty] in ["avg-logits"]
        model_name = get_model_name(path_name)
        
        # TODO make sure this works
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        sentence_pairs = create_sent_pairs(paragraphs) 
        test_set = BertPairDataset(tokenizer, sentence_pairs, labels=None)
        test_set.return_raw_text = sentence_transformers
        data_loader = DataLoader(test_set, batch_size=16, shuffle=False, pin_memory=True)

        # logits_loss = True means we return logits from forward pass
        model = BertStyleNN(enc_model_name=model_name, use_sentence_transformers=sentence_transformers, logits_loss=logits_loss)
        model.load_state_dict(torch.load(f"{path_name}-Best.pth"))
        model.to(device)
        
        # run inference!
        output = inference(model, data_loader, device)
        preds = indiv_preds(difficulty, output, logits_loss)
        all_preds.append(preds)
        all_outputs.append(output)
    
    if ENSEMBLE_METHODS[difficulty] == "maj-vote":
        ensemble_preds = ensemble_majority_voting(all_preds)
    else:
        # use majority voting for easy and medium
        ensemble_preds = ensemble_avg_outputs(difficulty, all_outputs, apply_sigmoid=logits_loss)
    return ensemble_preds

def indiv_preds(difficulty, outputs, logits_loss):
    # outputs is list for a single model (either logits or probs)
    # returns metrics, and PyTorch tensor with binary predictions
    if logits_loss:
        sigmoid = nn.Sigmoid()
        probs = sigmoid(outputs)
    else:
       probs = outputs
    
    # Test a range of thresholds to find the one that maximizes macro F1
    # use the best threshold to make final predictions
    preds = (probs >= BEST_THRESHOLDS[difficulty]).int()
    
    # calculate final metrics using the best threshold
    return preds

def ensemble_majority_voting(all_preds, majority_req=2):
    """
    Given preds (list of lists of binary predictions), implement majority voting where majority_req required to make
    1 prediction (in theory, make it easier to predict 1 due to class imbalance)
    """
    final_preds = (torch.sum(torch.stack(all_preds), dim=0) >= majority_req).int()
    return final_preds.numpy()

def ensemble_avg_outputs(difficulty, outputs, apply_sigmoid=False):
    """
    Given list of torch Tensors, calculate avg of outputs (optionally apply sigmoid) and then get predictions from that.
    Uses best threshold found from validation set

    Returns:
        preds (numpy.ndarray) with integer binary predictions
    """
    # first find avg of the outputs (whether logits or probs)
    avg_probs = torch.mean(torch.stack(outputs), dim=0)

    # optionally apply sigmoid to turn into probabilities
    if apply_sigmoid:
        sigmoid = nn.Sigmoid()
        avg_probs = sigmoid(avg_probs)
        avg_probs = torch.clamp(avg_probs, min=1e-6, max=1-1e-6)
    
    avg_probs_np = avg_probs.numpy()
    
    # use the best threshold (gotten from val set) to make final predictions
    preds = (avg_probs_np >= ENSEMBLE_THRESHOLDS[difficulty]).astype(int)
    return preds

def inference(model, data_loader, device):
    """
    Returns the direct output from the model (whether logits or probs is defined earlier in run_ensemble)
    Outputs are torch tensors on CPU!
    """
    model.eval()
    all_outputs = []
    
    with torch.no_grad():
        for batch in data_loader:
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
          
          all_outputs.append(outputs.detach().cpu())
    
    all_outputs = torch.cat(all_outputs)
    
    # return logits/probs (as torch.tensors on CPU)
    return all_outputs

def create_sent_pairs(paragraphs):
    """
    Args:
        paragraphs: List of sentences for a given problem
    Returns List[Tuple(str, str)], ready to be used in BertPairDataset
    """
    sentence_pairs = []
    for prob in paragraphs:
        # For each problem (list of sentences) and corresponding label (dict with 'changes' list)
        for i in range(len(prob) - 1):
            sentence_pairs.append((prob[i], prob[i + 1]))

    return sentence_pairs

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

@click.command()
@click.option('--dataset', default='multi-author-writing-style-analysis-2025/multi-author-writing-spot-check-20250503-training', help='The dataset to run predictions on (can point to a local directory).')
@click.option('--output', default=Path(get_output_directory(str(Path(__file__).parent))), help='The file where predictions should be written to.')
def main(dataset, output):
    """Main function that loads models, processes data, and writes predictions."""
    tira = Client()
    # Load the dataset
    input_df = tira.pd.inputs(dataset, formats=["multi-author-writing-style-analysis-problems"])
    logger.info(f"Successfully loaded dataset with {len(input_df)} problems")
    
    # process each difficulty subset
    for subtask in ["easy", "medium", "hard"]:
        subtask_problems = input_df[input_df["task"] == subtask]
        logger.info(f"Processing {len(subtask_problems)} problems for {subtask} subtask")
        run_problems(subtask, subtask_problems, Path(output))
    
    logger.info("All problems processed successfully")

if __name__ == '__main__':
    main()