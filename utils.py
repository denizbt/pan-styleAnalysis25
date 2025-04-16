"""
Utility, data augmentation functions for PAN 2025 Style Analysis Task
"""
import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm 
import nltk
nltk.download('punkt_tab')

import random
import pickle
import numpy as np

def sentence_pairs_paragraphs(paragraphs, num_pairs):
    """
    Randomly sample sentence pairs from a list of paragraphs. Ensures each sentence is used at most twice.
    
    Args:
        paragraphs (list): List of paragraph strings
        num_pairs (int): Number of pairs to create.
    
    Returns:
        list: List of sentence pairs (tuples)
    """
    paragraph_to_sentences = {}
    sentence_to_paragraph = {}
    all_sentences = []
    
    # tokenize sentences from paragraphs using NLTK
    # pair each sentence with its paragraph index
    for i, para in enumerate(paragraphs):
        sentences = nltk.sent_tokenize(para)
        paragraph_to_sentences[i] = sentences
        all_sentences.extend(sentences)
        for sentence in sentences:
          sentence_to_paragraph[sentence] = i
    
    print("paragraphs tokenized")
    # how many times each sentence was paired
    use_count = {sentence: 0 for sentence in all_sentences}
    
    pairs = []
    sentence_queue = []    
    while len(pairs) < num_pairs:
        # pair sentence from queue (if there exists one)
        if len(sentence_queue) > 0:        
          s1 = sentence_queue.pop()
        else:
          # otherwise, randomly sample a sentence
          s1 = random.choice(all_sentences)
          
        # identify paragraph that sentence is from
        p1 = sentence_to_paragraph[s1]
        
        # if this sentence already been used twice, don't use it
        if use_count[s1] >= 2:
          continue
        
        candidates = [s2 for s2 in all_sentences if sentence_to_paragraph[s2] != p1 and s1 != s2]        
        # if no candidates, skip this sentence
        if len(candidates) == 0:
          continue
        
        for i in range(2-use_count[s1]):
          if len(candidates) == 0:
            break
          
          s2 = random.choice(candidates)
          pairs.append((s1, s2))
        
          use_count[s1] += 1
          use_count[s2] += 1
          
          # add this sentence to be paired up to twice (if needed)
          if use_count[s2] < 2:
            sentence_queue.append(s2)
          candidates.remove(s2)
  
    pairs = pairs[:num_pairs]
    with open(f"new_sentence_pairs.pkl", "wb") as f:
      pickle.dump(pairs, f)
      
    return pairs

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def save_embeddings(data, split="train"):
  model_name = "sentence-transformers/all-MiniLM-L12-v2"
  # model_name = "princeton-nlp/unsup-simcse-roberta-base"
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

    with torch.no_grad():
      out1 = model(**input1)
      out2 = model(**input2)

    # # Perform pooling and normalize
    emb1 = mean_pooling(out1, input1['attention_mask'])
    emb2 = mean_pooling(out2, input2['attention_mask'])
    emb1 = F.normalize(emb1, p=2, dim=1)
    emb2 = F.normalize(emb2, p=2, dim=1)
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
  torch.save(sentence_embeddings, f"EXTRA_{file_path}_{split}.pt")

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

def read_labeled_data(dir):
  """
  Read in the data from directory that contains problems and labels.
  
  Args:
    dir: Path[str], path to the directory with .txt and .json files (dataset)
  
  Returns: 
    problems: List[List[str]], list of new line separated sentences for each doc
    labels: List[Dict[str, Any]]
  """
  problems = [] # list of lists, each list is a problem and contains the list of sentences
  labels = []
  files = os.listdir(dir)
  files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
  for file in files:
    with open(dir+"/"+file, "r") as f:
      if(file.endswith(".txt")):
        # a problem file
        problem = f.readlines()
        problems.append([p.strip() for p in problem]) # assumption: split based on periods, not newlines
      else:
        # a truth file
        truth = json.load(f)
        labels.append(truth)
  
  assert len(problems) == len(labels)
  return problems, labels

def pair_sentences_with_labels(problems, labels):
    """
    returns
    sentence_pairs: List[Tuple(str, str)]
    label_pairs: List[int \in {0, 1}]
    len(sentence_pairs) == len(label_pairs)
    """
    sentence_pairs = []
    label_pairs = []

    count = 1
    for prob, label in zip(problems, labels):
        # For each problem (list of sentences) and corresponding label (dict with 'changes' list)
        changes = label['changes']

        if len(prob) - 1 != len(changes):
            continue 
          # hard training problem-3207.txt is broken (extra new line I think, the number of sentence pairs does not match)

        for i in range(len(prob) - 1):
            sentence_pairs.append((prob[i], prob[i + 1]))
            label_pairs.append(changes[i])
        
        count += 1

    return sentence_pairs, label_pairs

def combine_paragraphs(probs, labels):
  """
  Args:
    probs: list[list[str]]
    labels: list[int], either {0, 1} representing style changes
    len(labels)+1 == len(probs)
  
  Returns:
    list[str] where each str is single author paragraph. each separate element is different author
  """
  paras = []
  for p in probs:
    comb = []
    curr_para = p[0]
    for i in range(len(p)-1):
      if labels[i] == 1:
        comb.append(curr_para)
        curr_para = ""
      
      curr_para += p[i+1]
    
    if curr_para != "":
      comb.append(curr_para)

    paras += comb
    
  return paras

def concat_balanced_dataset(extra_pairs_path, base_train_path, model_name):
  extra_pairs = torch.load(extra_pairs_path)
  base_pairs = torch.load(base_train_path)
  all_pairs = torch.concat([base_pairs, extra_pairs], dim=0)
  
  # all new pairs are style change pairs i.e. label 1
  all_labels = np.concatenate((np.load("train_labels.npy"), np.ones((extra_pairs.size(0),), dtype=np.int8)))
  
  print(f"labels {len(all_labels)}")
  print(f"embeddings {all_pairs.size()}")
  np.save("train_bal_labels.npy", all_labels)
  torch.save(all_pairs, f"{model_name}_bal_train.pt")

def save_style_change_pairs(new_pairs=95196):
  DATA_DIR = "pan24-data"
  train_easy_probs, train_easy_labels = read_labeled_data(f"{DATA_DIR}/easy/train")
  train_med_probs, train_med_labels = read_labeled_data(f"{DATA_DIR}/medium/train")
  train_hard_probs, train_hard_labels = read_labeled_data(f"{DATA_DIR}/hard/train")
 
  combined_probs = combine_paragraphs(train_easy_probs, train_easy_labels)
  combined_probs += combine_paragraphs(train_med_probs, train_med_labels)
  combined_probs += combine_paragraphs(train_hard_probs, train_hard_labels)
  
  print("paragraphs combined.")
  # 158,280 total sent pairs in 2025
  # 31,542 diff author sent pairs
  # need 95,196 NEW diff author sent pairs to balance out training set
  # each unique sentence added to dataset should be present in two pairs (both diff authors labels)
  pairs = sentence_pairs_paragraphs(combined_probs, num_pairs=new_pairs)

  print("pairs created.")  
  # create embeddings from pairs
  save_embeddings(pairs, "train")

if __name__ == "__main__":
  pass
    