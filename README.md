# Multi-Author Style Analysis
[PAN Lab Shared Task @ CLEF 2025](https://pan.webis.de/clef25/pan25-web/style-change-detection.html)

> :memo: **Notebook Paper:** In progress!

## Repository Structure
* data/
  * Contains PAN 2025 data for style analysis task. Download [here](https://zenodo.org/records/14891299).
* ensemble/
  * ensemble-cls.py &rarr; Functions to determine best ensemble on validation set using finetuned models. Supports majority voting, avg probabilities, avg logits methods.
  * models.py &rarr; BertStyleNN, BertPairDataset, StyleNN classes (imported in ensemble-cls.py). Compiled from training/bert-training.py and training/ffnn.py
* logs/
  * bert-trained/ &rarr; Training logs for all models used in final ensemble method (and others) 
  * Other log files from baseline/naive experimentation i.e. using static embeddings without fine-tuning on style analysis task.
* training/
  * bert-training.py &rarr; Fine-tuning code for encoder & binary classification head. Supports most HuggingFace encoder-only models (including BERT family) as well as many SentenceTransformers models. Comment in file specifies all models which definitely work.
  * ffnn.py &rarr; Defines FFNN used as binary classification head.
  * siamese.py &rarr; Siamese style network for fine-tuning embeddings. Did not work well (not used).

## Reproduction
In order to reproduce my exact results on the PAN shared task, you must download the state dictionaries for the fine-tuned models.

### Download Model Files
You can download fine-tuned model state dictionaries used in this submission to PAN 2025 directly from HuggingFace. You can view all available models [here](https://huggingface.co/denizbt/pan-style-analysis-models).

```
# Example downloading state dictionary for fine-tuned roberta-base to root directory.
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='denizbt/pan-style-analysis-models', filename='roberta-base.pth', local_dir='.')
```

or 

```
# Download all files from your repository to current directory
snapshot_download(
    repo_id='denizbt/pan-style-analysis-models',
    local_dir='.',
    local_dir_use_symlinks=False
)
```

<!-- ### **Alternative**: Run Docker Image Submission
TODO will it be possible to download?  -->

<!-- !!! dont want to make other repo public. def link to download docker image if possible -->

<!-- > :pushpin: [This repository](https://github.com/denizbt/pan-tira-submission) contains the Dockerfile and all necessary files that was used to make official submission to TIRA.io. Requires authentication token to run i.e can't be built and pushed by public. If you want to run my Docker image, use instructions above. -->

## Shared Task Results
coming soon!

## Train a BertStyleNN from Scratch
TODO

## Author
Deniz Bölöni-Turgut // db823@cornell.edu