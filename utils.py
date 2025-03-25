"""
Utility functions for PAN 2025 task
"""
import os
import json

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
