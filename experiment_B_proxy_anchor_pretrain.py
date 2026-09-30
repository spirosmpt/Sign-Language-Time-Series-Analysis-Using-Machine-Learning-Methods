
import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from pretraining.embeding_networks import HandEmbeddingMLP


class ProxyAnchorLoss(nn.Module):
    def __init__(self, num_classes, emb_dim, margin=0.1, alpha=32):
        super().__init__()
        self.proxies = nn.Parameter(torch.randn(num_classes, emb_dim))
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        self.num_classes = num_classes
        self.margin = margin
        self.alpha = alpha

    def forward(self, X, labels):
        P = F.normalize(self.proxies, dim=1)
        X = F.normalize(X, dim=1)
        cos = X @ P.T
        one_hot = torch.zeros(len(labels), self.num_classes, device=X.device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        pos_exp = torch.exp(-self.alpha * (cos - self.margin))
        neg_exp = torch.exp(self.alpha * (cos + self.margin))
        with_pos = (one_hot.sum(0) > 0)
        num_valid = with_pos.sum().clamp(min=1)
        P_sum = (one_hot * pos_exp).sum(0)
        N_sum = ((1 - one_hot) * neg_exp).sum(0)
        pos_term = torch.log(1 + P_sum)[with_pos].sum() / num_valid
        neg_term = torch.log(1 + N_sum).sum() / self.num_classes
        return pos_term + neg_term


class HandFrameDataset(Dataset):
    def __init__(self, X, y, hand_start=54, hand_stop=75):
        hand_feats = X[:, :, hand_start*3 : hand_stop*3]
        N, T, D = hand_feats.shape
        self.X = torch.tensor(hand_feats.reshape(N*T, D), dtype=torch.float32)
        self.y = torch.tensor(np.repeat(y, T), dtype=torch.long)
        valid = self.X.abs().sum(dim=1) > 0
        self.X = self.X[valid]
        self.y = self.y[valid]
        print(f"  Frames: {len(self.X)} valid / {N*T} total")

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X_train = np.load(os.path.join(args.data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(args.data_dir, "y_train.npy"))
    num_classes = len(np.unique(y_train))
    print(f"X_train: {X_train.shape}, classes: {num_classes}")

    dataset = HandFrameDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    encoder = HandEmbeddingMLP(input_dim=63, emb_dim=args.emb_dim).to(device)
    criterion = ProxyAnchorLoss(num_classes, args.emb_dim, args.margin, args.alpha).to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(criterion.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"HandEmbeddingMLP params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"\n{'Epoch':>6} | {'Loss':>8}")
    print("-" * 20)

    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        total_loss, n_batches = 0.0, 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(encoder(X_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()
        avg_loss = total_loss / n_batches
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(encoder.state_dict(), args.save_path)
        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6} | {avg_loss:>8.4f}")

    print(f"\nBest loss: {best_loss:.4f}")
    print(f" Encoder αποθηκεύτηκε: {args.save_path}")
    print(f"\n Επόμενο βήμα:")
    print(f"   python src/training/train.py --data_dir {args.data_dir} --encoder_ckpt {args.save_path} --hand_emb_dim {args.emb_dim} --seed 1 --epochs 50 --batch_size 32 --lr 3e-4 --wd 1e-4 --heads 8 --layers 4 --model_dim 256 --use_posenc")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   type=str, required=True)
    p.add_argument("--save_path",  type=str, default="outputs/proxy_encoder.pt")
    p.add_argument("--emb_dim",    type=int, default=64)
    p.add_argument("--epochs",     type=int, default=30)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--margin",     type=float, default=0.1)
    p.add_argument("--alpha",      type=float, default=32)
    train(p.parse_args())
