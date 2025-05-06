# TODO add ensemble model pipeline code here
# TODO need to use docker to upload our final system. include pth files

import argparse
import os
import json

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", type=str, help="INPUT DIR")
  parser.add_argument("-o", type=str, help="OUTPUT DIR")
  
  return parser.parse_args()

def extract_data(directory):
  problems = [] # list of lists, each list is a problem and contains the list of sentences
  labels = []
  files = os.listdir(directory)
  files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
  for file in files:
    #print(file)
    with open(directory+"/"+file, "r") as f:
      if(file.endswith(".txt")):
        # a problem file
        problem = f.readlines()
        problems.append(problem)
      else:
        # a truth file
        truth = json.load(f)
        labels.append(truth)
  
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

def data_creation(data_dir):
  """
  Returns:
    List[(s1, s2)], List[labels(int)]
  """
  
  easy_probs, easy_labels = extract_data(f"{data_dir}/easy/{s}")
  med_probs, med_labels = extract_data(f"{data_dir}/medium/{s}")
  hard_probs, hard_labels = extract_data(f"{data_dir}/hard/{s}")
  
  probs = easy_probs + med_probs + hard_probs
  labels = easy_labels + med_labels + hard_labels
  
  pairs, label_pairs = pair_sentences_with_labels(probs, labels)
  return pairs, label_pairs

if __name__ == "__main__":
  args = get_args()
  
  # pairs is List of tuples (s1, s2)
  # labels is List of int {0, 1} denoting whether style change or not
  pairs, labels = data_creation(args.i)