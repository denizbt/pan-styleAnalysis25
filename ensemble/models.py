import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sentence_transformers import SentenceTransformer

class BertPairDataset(Dataset):
  """
  Custom (PyTorch) dataset
  """
  def __init__(self, tokenizer, sent_pairs, labels, max_length=175):
    """
    Args:
      tokenizer: the BERT(+) tokenizer to use for the sentence pairs
      sent_pairs List[Tuple]: each element of list is tuple (sentence1, sentence2)
      labels List[int]: 0 or 1, where a 1 indicates style difference between pairs of sentences 
      len(sent_pairs) == len(labels)
    """
    self.tokenizer = tokenizer
    self.sent_pairs = sent_pairs
    self.labels = labels
    self.max_length = max_length
  
  def __len__(self):
    return len(self.labels)
  
  def __getitem__(self, idx):
    s1, s2 = self.sent_pairs[idx]
    label = self.labels[idx]
    
    # return non-tokenized text if sentence-transformers being used
    if self.return_raw_text:
      return {
          's1': s1,
          's2': s2,
          'labels': torch.tensor(label, dtype=torch.float)
      }
    
    # get tokenized values for both sentences in pair
    tok1 = self.tokenizer(
      s1,
      add_special_tokens=True,
      max_length=self.max_length,
      padding='max_length',
      truncation=True,
      return_tensors='pt'
    )
        
    tok2 = self.tokenizer(
      s2,
      add_special_tokens=True,
      max_length=self.max_length,
      padding='max_length',
      truncation=True,
      return_tensors='pt'
    )
    
    return {'input_ids1': tok1['input_ids'].squeeze(0),
          'attention_mask1': tok1['attention_mask'].squeeze(0),
          'input_ids2': tok2['input_ids'].squeeze(0),
          'attention_mask2': tok2['attention_mask'].squeeze(0),
          'labels': torch.tensor(label, dtype=torch.float)}

def mean_pooling(model_output, attention_mask):
  token_embeddings = model_output.last_hidden_state  # [batch_size, seq_len, hidden_dim]
  input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
  sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
  sum_mask = input_mask_expanded.sum(1)
  return sum_embeddings / sum_mask.clamp(min=1e-9)  # prevent division by zero

class BertStyleNN(nn.Module):
  """
  NN which uses BERT(etc.) encoder and separate MLP for classification.
  1. Uses self.encoder for independent feature extraction of two sentences.
  2. Concatenates the two embeddings, and passes through StyleNN (defined in mlp.py) for final classification.
  """
  def __init__(self, hidden_dims=[512, 256, 128, 64], output_dim=1, enc_model_name='roberta-base', pooling='mean', use_sentence_transformers=False):
    """
    Args:
      hidden_dims [List[int]]:
      output_dim [int]: final classification of the model is a single dimension
      pooling [str] in ['mean', 'cls]
      resume_training: If not None, contains path to encoder-only model state dict from which to resume training
    """
    super(BertStyleNN, self).__init__()  

    # check if it's a sentence transformers model
    # self.sentence_transformers = use_sentence_transformers
    if use_sentence_transformers:
      self.encoder = SentenceTransformer(enc_model_name)
      # self.encoder = SentenceTransformer(enc_model_name, trust_remote_code=True, model_kwargs={'default_task': 'text-matching'})
      with torch.no_grad():
        dummy_embedding = self.encoder.encode(["Hello"], convert_to_tensor=True)
        embedding_dim = dummy_embedding.shape[-1]
    else:
      self.encoder = AutoModel.from_pretrained(enc_model_name)  
      embedding_dim =  self.encoder.config.hidden_size

    self.pooling = pooling
    # input to MLP is the concatenation of the sentence pairs extracted
    self.mlp = StyleNN(input_dim=embedding_dim*2, hidden_dims=hidden_dims, output_dim=output_dim)
  
  def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
    # extract features from encoder independently on the two sentences
    if attention_mask1 is None:
      s1 = self.encoder.encode(input_ids1, convert_to_tensor=True, batch_size=16, show_progress_bar=False).to(next(self.parameters()).device)
      s2 = self.encoder.encode(input_ids2, convert_to_tensor=True, batch_size=16, show_progress_bar=False).to(next(self.parameters()).device)
    else:
      s1 = self.encoder(input_ids=input_ids1, attention_mask=attention_mask1)
      s2 = self.encoder(input_ids=input_ids2, attention_mask=attention_mask2)
      
      if self.pooling == 'mean':
        s1 = mean_pooling(s1, attention_mask1)
        s2 = mean_pooling(s2, attention_mask2)
      else:
        # pooler_output takes [CLS] hidden layer vector
        s1 = s1.pooler_output if s1.pooler_output is not None else s1.last_hidden_state[:, 0, :]
        s2 = s2.pooler_output if s2.pooler_output is not None else s2.last_hidden_state[:, 0, :]
    
    # concatenate features from sentece pairs to pass into MLP for classification
    # using BCEWithLogitsLoss, so no sigmoid
    concat = torch.cat((s1, s2), dim=1)
    logits = self.mlp(concat)
    return logits

class StyleNN(nn.Module):
    """
    MLP for binary classification of sentences for same author or not.
    """
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64], output_dim=1, p=0.4, apply_sigmoid=True):
        """
        Args:
          p: dropout rate
          apply_sigmoid: True if model should apply sigmoid as last step in forward()
        """
        super(StyleNN, self).__init__()
        layers = []
        
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.Dropout(p))
        
        # define all hidden layers
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            layers.append(nn.Dropout(p))
        
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.model = nn.Sequential(*layers)
        
        self.sigmoid = nn.Sigmoid()
        self.apply_sigmoid = apply_sigmoid
        
    def forward(self, x):
      x = self.model(x)
      
      # apply sigmoid if not using BCEWithLogitsLoss
      if self.apply_sigmoid:
        x = self.sigmoid(x)
      
      return x
