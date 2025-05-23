# Multi-Author Style Analysis (PAN Lab Shared Task @ CLEF 2025)
[Official Shared Task Page](https://pan.webis.de/clef25/pan25-web/style-change-detection.html)

> :memo: **Notebook Paper:** In progress!

## Author
Deniz Bölöni-Turgut // db823@cornell.edu

## Repository Structure
* data/
  * Contains PAN 2025 data for style analysis task. Download [here](https://zenodo.org/records/14891299).
* ensemble/
  * ensemble-cls.py &rarr; Functions to determine best ensemble on validation set using finetuned models. Supports majority voting, avg probabilities, avg logits methods.
  * models.py &rarr; BertStyleNN, BertPairDataset, StyleNN classes (imported in ensemble-cls.py). Compiled from training/bert-training.py and training/mlp.py
* logs/
  * bert-trained/ &rarr; Training logs for all models used in final ensemble method (and others) 
  * Other log files from baseline/naive experimentation i.e. using static embeddings without fine-tuning on style analysis task.
* submission/
  * All files packed into Docker container for final submission to TIRA.io. Includes Dockerfile.
* training/
  * bert-training.py &rarr; Fine-tuning code for encoder & binary classification head. Supports most HuggingFace encoder-only models (including BERT family) as well as many SentenceTransformers models. Comment in file specifies all models which definitely work.
  * mlp.py &rarr; Defines FFNN used as binary classification head.
  * siamese.py &rarr; Siamese style network for fine-tuning embeddings. Did not work well (not used).

## Methodology
TODO

## Results
TODO (awaiting leaderboard results on test set)

## Reproduction
TODO (upload .pth files somehow)