# TODO finish the final pipeline code here (for system submission)
# TODO need to use docker to upload our final system

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

if __name__ == "__main__":
  args = get_args()
  problems, labels = extract_data(args.i)