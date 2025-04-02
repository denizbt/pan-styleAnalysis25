import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm

from utils import *

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
    
def main():
  # train_sent, train_pairs, train_labels = get_data("train")
  val_problems, val_labels = read_labeled_data("val")
  
  cosine_sim_classification(val_problems, val_labels)
  

if __name__ == "__main__":
  main()