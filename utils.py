"""
Utility functions for PAN 2025 task
"""
import os
import json
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm 

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

## read in data from dir with txt and json 
def read_labeled_data(dir):
  """
  returns 
  problems: List[List[str]], list of new line separated sentences for each doc
  labels: List[Dict[str, Any]]
  """
  problems = [] # list of lists, each list is a problem and contains the list of sentences
  labels = []
  files = os.listdir(dir)
  files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
  for file in files:
    #print(file)
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
