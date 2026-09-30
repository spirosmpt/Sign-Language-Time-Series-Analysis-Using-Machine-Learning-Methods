"""
Experiment C: Hyperformer για WLASL-100
Απλοποιημένη έκδοση για 4GB GPU.

Τρόπος χρήσης:
  python experiment_C_hyperformer_wlasl.py \
      --data_dir data/WLASL_npy_dataset_100_split \
      --epochs 50 --batch_size 16
"""

import os, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────
# MediaPipe graph (75 joints)
# ─────────────────────────────────────────────
def get_mediapipe_adj(V=75):
    A = np.zeros((V, V))
    # Pose
    for a, b in [(0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
                 (9,10),(11,12),(11,13),(13,15),(12,14),(14,16),
                 (11,23),(12,24),(23,24),(23,25),(25,27),(27,29),
                 (24,26),(26,28),(28,30),(15,17),(15,19),(16,18),(16,20)]:
        if a < V and b < V: A[a,b] = A[b,a] = 1
    # Hand edges (same topology for both hands)
    hand_e = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
              (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
              (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
    for off in [33, 54]:  # left hand, right hand
        for a, b in hand_e:
            if a+off < V and b+off < V: A[a+off,b+off] = A[b+off,a+off] = 1
    # Wrist connections
    if 15 < V and 33 < V: A[15,33] = A[33,15] = 1
    if 16 < V and 54 < V: A[16,54] = A[54,16] = 1
    np.fill_diagonal(A, 1)
    return A


def compute_hops(A, max_hops=6):
    """Shortest path distance (capped at max_hops)."""
    V = A.shape[0]
    hops = np.full((V, V), max_hops, dtype=np.int64)
    np.fill_diagonal(hops, 0)
    curr = (A > 0).astype(float)
    for k in range(1, max_hops + 1):
        mask = (curr > 0) & (hops > k)
        hops[mask] = k
        curr = curr @ (A > 0).astype(float)
    return hops


# ─────────────────────────────────────────────
# HyperSA Layer
# ─────────────────────────────────────────────
class HyperSA(nn.Module):
    """
    Hypergraph Self-Attention (Paper 3).
    Κάθε joint προσέχει:
    (a) άλλα joints (vanilla attention)
    (b) την ομάδα (hyperedge) που ανήκει
    (c) k-hop positional bias
    """
    def __init__(self, dim, num_heads, num_joints, num_groups, hops):
        super().__init__()
        assert dim % num_heads == 0
        self.h = num_heads
        self.d = dim // num_heads
        self.scale = self.d ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)

        # Hyperedge partition
        self.partition = nn.Parameter(torch.randn(num_joints, num_groups) * 0.02)
        self.grp_proj  = nn.Linear(dim, dim, bias=False)

        # K-hop RPE: ένα scalar bias ανά hop distance (πολύ ελαφρύ)
        max_hop = int(hops.max()) + 1
        self.hop_bias = nn.Parameter(torch.zeros(self.h, max_hop))
        self.register_buffer("hops", torch.tensor(hops, dtype=torch.long))

    def forward(self, x):
        # x: (B, V, C)
        B, V, C = x.shape
        H, d = self.h, self.d

        qkv = self.qkv(x).reshape(B, V, 3, H, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, V, d)

        # (a) Joint-to-joint
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, V, V)

        # (c) K-hop bias (scalar per hop, shared across heads dimension)
        hop_b = self.hop_bias[:, self.hops]  # (H, V, V)
        attn = attn + hop_b.unsqueeze(0)

        # (b) Hyperedge attention
        H_soft = F.softmax(self.partition, dim=1)  # (V, G)
        grp = torch.einsum("vg,bvc->bgc", H_soft, x)     # (B, G, C)
        grp = self.grp_proj(grp)                           # (B, G, C)
        aug = torch.einsum("vg,bgc->bvc", H_soft, grp)    # (B, V, C)
        aug_k = aug.reshape(B, V, H, d).permute(0, 2, 1, 3)  # (B, H, V, d)
        attn = attn + (q @ aug_k.transpose(-2, -1)) * self.scale

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, V, C)
        return self.proj(out)


# ─────────────────────────────────────────────
# Temporal Conv Block
# ─────────────────────────────────────────────
class TemporalConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.BatchNorm1d(dim),
            nn.GELU(),
        )

    def forward(self, x):
        # x: (B, T, C)
        return x + self.conv(x.permute(0,2,1)).permute(0,2,1)


# ─────────────────────────────────────────────
# Hyperformer Block
# ─────────────────────────────────────────────
class HyperformerBlock(nn.Module):
    def __init__(self, dim, num_heads, num_joints, num_groups, hops, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.hyper = HyperSA(dim, num_heads, num_joints, num_groups, hops)
        self.norm2 = nn.LayerNorm(dim)
        self.temp  = TemporalConv(dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, V, C)
        B, T, V, C = x.shape

        # Spatial: process all frames independently
        xf = x.reshape(B*T, V, C)
        xf = xf + self.drop(self.hyper(self.norm1(xf)))
        x  = xf.reshape(B, T, V, C)

        # Temporal: process each joint independently
        xj = x.permute(0,2,1,3).reshape(B*V, T, C)
        xj = self.temp(self.norm2(xj))
        x  = xj.reshape(B, V, T, C).permute(0,2,1,3)

        return x


# ─────────────────────────────────────────────
# Full Model
# ─────────────────────────────────────────────
class HyperformerWLASL(nn.Module):
    def __init__(self, num_joints=75, num_classes=100,
                 dim=64, num_heads=4, num_layers=4,
                 num_groups=6, dropout=0.1):
        super().__init__()

        A    = get_mediapipe_adj(num_joints)
        hops = compute_hops(A)

        self.input_proj = nn.Linear(3, dim)
        self.pos_emb    = nn.Parameter(torch.zeros(1, 80, 1, dim))

        self.layers = nn.ModuleList([
            HyperformerBlock(dim, num_heads, num_joints, num_groups, hops, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        # x: (B, T, 225)
        B, T, F = x.shape
        V = F // 3
        x = x.reshape(B, T, V, 3)
        x = self.input_proj(x)          # (B, T, V, dim)
        x = x + self.pos_emb[:, :T]
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x).mean(dim=(1, 2))  # global avg pool
        return self.head(x)


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class WLASLDataset(Dataset):
    def __init__(self, split, data_dir):
        self.X = np.load(os.path.join(data_dir, f"X_{split}.npy")).astype(np.float32)
        self.y = np.load(os.path.join(data_dir, f"y_{split}.npy")).astype(np.int64)
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), int(self.y[i])


# ─────────────────────────────────────────────
# Eval helper
# ─────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    c1 = c5 = c10 = n = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            n  += y.size(0)
            c1 += (logits.argmax(-1) == y).sum().item()
            top5  = torch.topk(logits, k=min(5,  logits.size(1))).indices
            top10 = torch.topk(logits, k=min(10, logits.size(1))).indices
            c5  += (top5  == y.unsqueeze(1)).any(1).sum().item()
            c10 += (top10 == y.unsqueeze(1)).any(1).sum().item()
    return c1/n, c5/n, c10/n


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
def train(args):
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds_tr = WLASLDataset("train", args.data_dir)
    ds_va = WLASLDataset("val",   args.data_dir)
    ds_te = WLASLDataset("test",  args.data_dir)

    dl_tr = DataLoader(ds_tr, args.batch_size, shuffle=True,  num_workers=0)
    dl_va = DataLoader(ds_va, args.batch_size, shuffle=False, num_workers=0)
    dl_te = DataLoader(ds_te, args.batch_size, shuffle=False, num_workers=0)

    model = HyperformerWLASL(
        num_joints=75, num_classes=100,
        dim=args.dim, num_heads=args.num_heads,
        num_layers=args.num_layers, num_groups=args.num_groups,
        dropout=args.dropout,
    ).to(device)

    n_p = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Hyperformer parameters: {n_p:.2f}M")

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit  = nn.CrossEntropyLoss()

    best_val, best_state = 0.0, None

    print(f"\n{'Ep':>4} | {'Loss':>8} | {'Tr@1':>6} | {'Va@1':>6} | {'Va@5':>6} | {'Va@10':>7} | {'t':>5}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        model.train(); t0 = time.time()
        tl = tc = tn = 0
        for X, y in dl_tr:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            logits = model(X)           # μόνο ΜΙΑ φορά
            loss = crit(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * X.size(0)
            tc += (logits.detach().argmax(-1) == y).sum().item()
            tn += X.size(0)
        sched.step()

        v1, v5, v10 = evaluate(model, dl_va, device)
        if v1 > best_val:
            best_val = v1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({"model": model.state_dict()}, "checkpoints/hyperformer_best.pt")

        print(f"{epoch:>4} | {tl/tn:>8.4f} | {tc/tn:>6.3f} | {v1:>6.3f} | {v5:>6.3f} | {v10:>7.3f} | {time.time()-t0:>4.1f}s")

    if best_state:
        model.load_state_dict(best_state)
    t1, t5, t10 = evaluate(model, dl_te, device)

    print(f"\n{'='*45}")
    print(f"TEST RESULTS — Experiment C (Hyperformer)")
    print(f"{'='*45}")
    print(f"  Top-1:  {t1:.4f}  ({t1*100:.1f}%)")
    print(f"  Top-5:  {t5:.4f}  ({t5*100:.1f}%)")
    print(f"  Top-10: {t10:.4f}  ({t10*100:.1f}%)")
    print(f"{'='*45}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   type=str, required=True)
    p.add_argument("--epochs",     type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--dim",        type=int, default=64)
    p.add_argument("--num_heads",  type=int, default=4)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_groups", type=int, default=6)
    p.add_argument("--dropout",    type=float, default=0.1)
    train(p.parse_args())
