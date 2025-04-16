import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from scipy.spatial.distance import cosine

import numpy as np
import argparse
import json
from tqdm import tqdm

from sklearn.metrics import f1_score

# mlp.py (file I wrote)
from mlp import SentPairDataset, StyleNN, val_mlp

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--embedding-type", type=str, default="all-MiniLM-L12-v2"
  )
  
  return parser.parse_args()

class SiameseNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims=[350, 300], output_dim=256):
        """
        Siamese Network implementation
        """
        super(SiameseNetwork, self).__init__()
        
        # define all the layers of the network
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.2)) # for regularization
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # create sequential
        self.seq = nn.Sequential(*layers)
    
    def forward_one(self, x):
        """Forward pass for a single input"""
        return self.seq(x)
    
    def forward(self, x1, x2):
        """Forward pass for a pair of inputs"""
        o1 = self.forward_one(x1)
        o2 = self.forward_one(x2)
        return o1, o2

class SiameseDataset(Dataset):
    def __init__(self, embeddings_path, labels):
        # shape of embeddings_pairs: (total # of pairs) x 2 x (embedding dim)
        self.embeddings = torch.load(embeddings_path)
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Get the pair of embeddings directly
        pair = self.embeddings[idx]
        
        # Extract the two embeddings from the pair
        e1 = pair[0].float()
        e2 = pair[1].float()
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return e1, e2, label

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, temperature=0.25):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(self, output1, output2, label):
        # scale distances by temperature (lower temperature = sharper differences)
        euclidean_dist = F.pairwise_distance(output1, output2) / self.temperature
        
        loss = torch.mean(
            (1-label) * torch.pow(euclidean_dist, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_dist, min=0.0), 2)
        )
        return loss

def train_siamese(train_embedding_path, train_labels, val_embeddings_path, val_labels,
                          batch_size=64, epochs=15, patience=3):
    """
    Train the Siamese network
    """
    device='cuda' if torch.cuda.is_available() else 'cpu'
    
    # create dataset and dataloader   
    train_set = SiameseDataset(train_embedding_path, train_labels)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    if val_embeddings_path is not None and val_labels is not None:
        val_set = SiameseDataset(val_embeddings_path, val_labels)
        val_loader = DataLoader(val_set, batch_size=batch_size)

    # initialize model, loss, and optimizer
    embedding_dim = train_set.embeddings.size(2)
    model = SiameseNetwork(input_dim=embedding_dim).to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.AdamW(model.parameters())
    
    # training loop
    train_losses = []
    val_losses = []
    patience_counter = 0
    best_epoch = 0
    best_val_loss = float('inf')
    best_model_state = None
    for e in tqdm(range(epochs), desc="Epochs", position=0):
        model.train()
        running_loss = 0.0
        
        for embedding1, embedding2, label in tqdm(train_loader, desc=f"train batches (epoch {e+1})", position=1, leave=False):
            embedding1, embedding2, label = embedding1.to(device), embedding2.to(device), label.to(device)
    
            optimizer.zero_grad()
            output1, output2 = model(embedding1, embedding2)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)       

        # run validation
        val_epoch_loss = val_siamese(model, criterion, val_loader, device)
        val_losses.append(val_epoch_loss)

        # check for early stopping: curr val loss > prev. epoch
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            best_epoch = e+1
        else:
            patience_counter += 1
            
        print(f"\nEpoch {e+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}")
            
        if patience_counter >= patience:
            print(f"early stopping triggered after {e+1} epochs.")
            break

    # load best model epoch to return
    print(f"using best epoch {best_epoch} of the siamese NN")
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), "siamese_model.pth")
    return model, train_losses, val_losses

def val_siamese(model, criterion, val_loader, device):
    model.eval()
    val_running_loss = 0.0
    
    with torch.no_grad():
        for embedding1, embedding2, label in val_loader:
            embedding1, embedding2, label = embedding1.to(device), embedding2.to(device), label.to(device)
            
            output1, output2 = model(embedding1, embedding2)
            loss = criterion(output1, output2, label)
            
            val_running_loss += loss.item()
    
    val_epoch_loss = val_running_loss / len(val_loader)
    return val_epoch_loss

def extract_embeddings(model, embedding_dataset, batch_size=64, concat=True):
    """
    Extract new embeddings using single path from trained Siamese network.
    Returns concatenated, & normalized embeddings, ready for MLP classification
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    
    dataloader = DataLoader(embedding_dataset, batch_size=batch_size, shuffle=False)
    processed_embeddings = []
    
    with torch.no_grad():
        for embedding1, embedding2, _ in dataloader:
            embedding1, embedding2 = embedding1.to(device), embedding2.to(device)
            
            # extract and normalize the embeddings
            o1, o2 = model.forward(embedding1, embedding2)
            o1, o2 = F.normalize(o1, p=2), F.normalize(o2, p=2)

            if concat:
                # concatenate the embeddings for input to MLP
                combined = torch.cat([o1, o2], dim=1)
            else:
                # do not concatenate
                combined = torch.stack([o1, o2], dim=1)
                
            processed_embeddings.append(combined.cpu())
    
    return torch.vstack(processed_embeddings)

def mlp_classification(train_embeddings, train_labels, val_embeddings, val_labels, batch_size=64, epochs=10, patience=5):
    device= 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # create datasets, dataloaders
    train_set = SentPairDataset(train_embeddings, train_labels, direct_pass=True)
    val_set = SentPairDataset(val_embeddings, val_labels, direct_pass=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    embedding_dim = train_set.embeddings.size(1)    
    model = StyleNN(input_dim=embedding_dim)
    model.to(device)
  
    optimizer = optim.AdamW(model.parameters()) # default lr=0.001
    criterion = nn.BCELoss()

    best_metrics = {'f1': -1}
    patience_counter = 0
    best_epoch = 0

    epochs = 3
    for e in tqdm(range(epochs), desc="Epochs", position=0):
        train_running_loss = 0
        for inputs, labels in tqdm(train_loader, desc=f"train batches (epoch {e+1})", position=1, leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            model.train()
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()

            # save model after every training epoch
            torch.save(model.state_dict(), f"BAL_{args.embedding_type}_pipeline_{e}.pth")  

            avg_train_loss = train_running_loss / len(train_loader)
            metrics, avg_val_loss, _ = val_mlp(model, val_loader, criterion, device)
            print(f"\nepoch {e}\ntraining loss: {avg_train_loss:.4f}\nval loss: {avg_val_loss:.4f}")
            print(f"val metrics: {metrics}\n")

            if metrics['f1'] > best_metrics['f1']:
                best_metrics = metrics
                patience_counter = 0
                best_epoch = e+1
            else:
                patience_counter += 1
            
            # Early stopping condition: if patience exceeds the limit, stop training
            if patience_counter >= patience:
                print(f"early stopping triggered after {e+1} epochs.")
                break
    
    with open(f"BAL_siamese_mlp_val_metrics.json", "w+") as f:
        best_metrics = {k: float(v) if hasattr(v, 'item') else v for k, v in best_metrics.items()}
        json.dump(best_metrics, f)
    
    print(f"training over! best epoch was {best_epoch}")

def cosine_sim_classification(embeddings, labels):
    thresholds = [0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98]
    predictions = [[] for _ in range(len(thresholds))] # list of predictions
    
    for i in range(embeddings.size(0)):
        e1 = embeddings[i][0].squeeze()
        e2 = embeddings[i][1].squeeze()
        # print(f"e1 {e1.size()}")
        cosine_sim = cosine(e1, e2)
        
        for j in range(len(thresholds)):
            if cosine_sim > thresholds[j]:
                # no style change
                predictions[j].append(0)
            else:
                # style change
                predictions[j].append(1)
    
  
    print(f"cosine class performance:\n")
    for i in range(len(thresholds)):
        macro_f1 = f1_score(labels, predictions[i], average='macro')
        print(f"{thresholds[i]} Macro F1: {macro_f1}")

def main(args):
    # load saved labels (numpy array of changes)
    # invert labels such that it's 1 if same author, 0 if different author
    # OG data has 1 for style change, 0 for no style change
    train_labels = (np.load("train_bal_labels.npy")) ^ 1
    val_labels = (np.load("val_labels.npy")) ^ 1
    # print(f"same author {sum(train_labels)}, diff author {len(train_labels)-sum(train_labels)}")

    # load saved sentence embeddings
    train_path = args.embedding_type+"_bal_train.pt"
    val_path = args.embedding_type+"_val.pt"

    # train Siamese network
    model, train_losses, val_losses = train_siamese(
        train_path, train_labels,
        val_path, val_labels,
        epochs=10
    )
    
    ## TODO remove
    # model = SiameseNetwork(input_dim=384)
    # model.load_state_dict(torch.load("siamese_model.pth"))
    ### end remove

    # extract features for train & val embeddings
    val_set = SiameseDataset(val_path, val_labels)
    train_set = SiameseDataset(train_path, train_labels)
    siamese_val = extract_embeddings(model, val_set, concat=False)
    siamese_train = extract_embeddings(model, train_set, concat=False)
   
    # perform final classification with MLP, switch labels back before sending
    train_labels = (np.load("train_bal_labels.npy")) ^ 1
    val_labels = (np.load("val_labels.npy")) ^ 1

    mlp_classification(siamese_train, train_labels, siamese_val, val_labels)
    # cosine_sim_classification(siamese_train, train_labels)    

    # TODO check if i need to flip the labels (for final classification), i think not?

if __name__ == "__main__":
    args = get_args()
    main(args)