"""
Model training script for Forest Fire Spread Prediction
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import yaml
from tqdm import tqdm
import logging
from src.utils import load_config, calculate_iou, calculate_f1_score, setup_device

# --- Model Definitions ---
class SimpleTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, N, C)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class TransformerCAHybrid(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers, num_heads, ca_iterations, img_size):
        super().__init__()
        self.img_size = img_size
        self.embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.transformer_blocks = nn.ModuleList([
            SimpleTransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.ca_iterations = ca_iterations
        self.ca_update = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.out = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.embed(x)  # (B, hidden_dim, H, W)
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # (B, N, C)
        for block in self.transformer_blocks:
            x_flat = block(x_flat)
        x = x_flat.transpose(1, 2).reshape(B, C, H, W)
        # Cellular Automata update
        for _ in range(self.ca_iterations):
            x = x + torch.tanh(self.ca_update(x))
        out = torch.sigmoid(self.out(x))  # (B, 1, H, W)
        return out

# --- Dataset ---
class FireDataset(Dataset):
    def __init__(self, npz_files):
        self.files = npz_files
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        x = data['input_tensor'].astype(np.float32)
        y = data['fire_history'][-1].astype(np.float32)  # Use last day as label
        return torch.tensor(x), torch.tensor(y)

# --- Training ---
def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    bce = nn.BCELoss(reduction='none')(pred, target)
    pt = torch.exp(-bce)
    loss = alpha * (1 - pt) ** gamma * bce
    return loss.mean()

def train():
    config = load_config()
    device = setup_device()
    batch_size = config['training']['batch_size']
    lr = config['training']['learning_rate']
    epochs = config['training']['epochs']
    ca_iterations = config['model']['ca_iterations']
    img_size = tuple(config['data']['input_size'])
    in_channels = config['model']['input_channels']
    hidden_dim = config['model']['hidden_dim']
    num_layers = config['model']['num_layers']
    num_heads = config['model']['num_heads']

    # Find all preprocessed .npz files
    data_dir = config['paths']['data_dir']
    npz_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]
    dataset = FireDataset(npz_files)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = TransformerCAHybrid(in_channels, hidden_dim, num_layers, num_heads, ca_iterations, img_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_iou = 0
    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            out = out.squeeze(1)
            loss = focal_loss(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch+1} Loss: {np.mean(losses):.4f}")
        # Validation (simple, on train set)
        model.eval()
        ious, f1s = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb).squeeze(1).cpu().numpy()
                yb = yb.cpu().numpy()
                for p, t in zip(out, yb):
                    pred_bin = (p > 0.5).astype(np.uint8)
                    t_bin = (t > 0.5).astype(np.uint8)
                    ious.append(calculate_iou(pred_bin, t_bin))
                    f1s.append(calculate_f1_score(pred_bin, t_bin))
        mean_iou = np.mean(ious)
        mean_f1 = np.mean(f1s)
        print(f"Validation IoU: {mean_iou:.3f} | F1: {mean_f1:.3f}")
        # Save best model
        if mean_iou > best_iou:
            best_iou = mean_iou
            os.makedirs(config['paths']['models_dir'], exist_ok=True)
            torch.save(model.state_dict(), os.path.join(config['paths']['models_dir'], 'best_model.pt'))
            print("Best model saved!")

if __name__ == "__main__":
    train() 