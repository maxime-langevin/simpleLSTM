import os
import re
import json
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from rdkit import Chem
import numpy as np

# --- Device ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# --- Tokenization ---
def replace_halogen(smiles):
    return smiles.replace("Cl", "L").replace("Br", "R")

def tokenize_smiles(smiles):
    regex = '(\[[^\[\]]{1,6}\])'
    smiles = replace_halogen(smiles)
    char_list = re.split(regex, smiles)
    tokens = []
    for segment in char_list:
        if segment.startswith('['):
            tokens.append(segment)
        else:
            tokens.extend(list(segment))
    tokens.append('EOS')
    return tokens

# --- Vocabulary ---
class Vocabulary:
    def __init__(self, smiles_list):
        tokens = set()
        for smi in smiles_list:
            tokens.update(tokenize_smiles(smi))
        self.special = ['EOS', 'GO']
        self.tokens = sorted(tokens.difference(set(self.special))) + self.special
        self.stoi = {t: i for i, t in enumerate(self.tokens)}
        self.itos = {i: t for t, i in self.stoi.items()}
        print(self.tokens)
        print(self.stoi)
        print(self.itos)

    def encode(self, tokens):
        return [self.stoi[t] for t in tokens if t in self.stoi]

    def decode(self, indices):
        tokens = [self.itos[i] for i in indices]
        if 'EOS' in tokens:
            tokens = tokens[:tokens.index('EOS')]
        return ''.join(tokens).replace('L', 'Cl').replace('R', 'Br')

    def __len__(self):
        return len(self.tokens)

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump({'stoi': self.stoi, 'itos': self.itos}, f)
        print(f"✅ Vocabulary saved to: {filepath}")

# --- Dataset ---
class SMILESDataset(Dataset):
    def __init__(self, smiles, vocab):
        self.vocab = vocab
        self.smiles = smiles

    def __getitem__(self, idx):
        tokens = tokenize_smiles(self.smiles[idx])
        ids = self.vocab.encode(tokens)
        x = torch.tensor([self.vocab.stoi['GO']] + ids[:-1], dtype=torch.long)
        y = torch.tensor(ids, dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.smiles)

def collate_fn(batch):
    xs, ys = zip(*batch)
    max_len = max(len(seq) for seq in ys)
    padded_x = torch.zeros(len(xs), max_len, dtype=torch.long)
    padded_y = torch.zeros(len(xs), max_len, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        padded_x[i, :len(x)] = x
        padded_y[i, :len(y)] = y
    return padded_x.to(device), padded_y.to(device)

# --- Model ---
class LSTMGen(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, hidden_dim=512, num_layers=3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        nn.init.xavier_uniform_(self.emb.weight)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.emb(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden

    def sample(self, vocab, max_len=100):
        self.eval()
        idx = torch.tensor([[vocab.stoi['GO']]], device=device)
        generated = []
        hidden = None
        with torch.no_grad():
            for _ in range(max_len):
                logits, hidden = self.forward(idx, hidden)
                probs = torch.softmax(logits[:, -1], dim=-1)
                if torch.isnan(probs).any():
                    break
                next_token = torch.multinomial(probs, num_samples=1)
                token_id = next_token.item()
                if token_id >= len(vocab.itos):
                    token_id = vocab.stoi['EOS']
                token = vocab.itos[token_id]
                if token == 'EOS':
                    break
                generated.append(token_id)
                idx = next_token
        return vocab.decode(generated)

# --- SMILES validity check ---
def is_valid_smiles(smiles):
    return Chem.MolFromSmiles(smiles) is not None

# --- Train and monitor ---
def train_one_epoch(file_path, batch_size=64, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)

    with open(file_path, 'r') as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    random.shuffle(smiles_list)

    vocab = Vocabulary(smiles_list)
    dataset = SMILESDataset(smiles_list, vocab)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = LSTMGen(len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    losses, perplexities, validities = [], [], []
    num_batches = len(loader)
    monitor_steps = np.linspace(0, num_batches - 1, 100, dtype=int)

    model.train()
    for i, (x, y) in enumerate(tqdm(loader, desc="Training 1 epoch")):
        logits, _ = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i in monitor_steps:
            with torch.no_grad():
                perplexity = torch.exp(loss).item()
                samples = [model.sample(vocab) for _ in range(50)]
                validity = np.mean([is_valid_smiles(smi) for smi in samples])
            losses.append(loss.item())
            perplexities.append(perplexity)
            validities.append(validity)

    # --- Save model and vocabulary ---
    model_path = os.path.join(checkpoint_dir, "lstm_gen.pt")
    vocab_path = os.path.join(checkpoint_dir, "vocab.json")
    torch.save(model.state_dict(), model_path)
    vocab.save(vocab_path)
    print(f"✅ Model saved to: {model_path}")

    # --- Save metrics to CSV ---
    metrics_path = os.path.join(checkpoint_dir, "training_metrics.csv")
    with open(metrics_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["step", "loss", "perplexity", "validity"])
        for step, (l, p, v) in enumerate(zip(losses, perplexities, validities)):
            writer.writerow([step, l, p, v])
    print(f"Training metrics saved to: {metrics_path}")

    # --- Plotting ---
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    axs[0].plot(losses)
    axs[0].set_title("Loss per step")
    axs[1].plot(perplexities)
    axs[1].set_title("Perplexity per step")
    axs[2].plot(validities)
    axs[2].set_title("Average SMILES Validity per step")
    for ax in axs:
        ax.set_xlabel("Checkpoint (1/100th epoch)")
    plt.tight_layout()
    plot_path = os.path.join(checkpoint_dir, "training_monitoring.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Plot saved to: {plot_path}")

if __name__ == "__main__":
    train_one_epoch("data/chembl28.smi", batch_size=64)
