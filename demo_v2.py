
import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.training.models import ModelConfig, SignTransformer

DATA_DIR = "data/WLASL_npy_dataset_100_split"
CKPT = "logs/20260504_210805/best.pt"
ENCODER_CKPT = ["outputs/proxy_encoder.pt"]

BODY_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
]
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

def load_model(device, num_classes, T, F):
    cfg = ModelConfig(
        feat_dim=F, model_dim=256, heads=8, layers=4, dropout=0.1,
        use_posenc=True, num_classes=num_classes, hand_emb_dim=64,
        encoder_ckpt=ENCODER_CKPT, freeze_encoder=False,
        use_hand_encoder=True, encoder_fusion="mean", encoder_dropout=0.0,
    )
    model = SignTransformer(cfg, T=T).to(device)
    ckpt = torch.load(CKPT, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model

def setup_skeleton_ax(ax, title, color='#3498db'):
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlim(-0.2, 1.2); ax.set_ylim(1.2, -0.2)
    ax.set_aspect('equal'); ax.axis('off')
    scat = ax.scatter([], [], s=8, c='#2c3e50')
    lines = [ax.plot([], [], lw=1.5, c=color)[0] for _ in
             range(len(BODY_CONNECTIONS) + 2 * len(HAND_CONNECTIONS))]
    return scat, lines

def draw_frame(pose_frame, scat, lines):
    P = pose_frame
    valid = ~np.all(P == 0, axis=1)
    scat.set_offsets(P[valid][:, :2])
    li = 0
    for a, b in BODY_CONNECTIONS:
        if valid[a] and valid[b]:
            lines[li].set_data([P[a,0], P[b,0]], [P[a,1], P[b,1]])
        else:
            lines[li].set_data([], [])
        li += 1
    for offset in (33, 54):
        for a, b in HAND_CONNECTIONS:
            A, B = a + offset, b + offset
            if valid[A] and valid[B]:
                lines[li].set_data([P[A,0], P[B,0]], [P[A,1], P[B,1]])
            else:
                lines[li].set_data([], [])
            li += 1
    return [scat] + lines

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--num_samples", type=int, default=1)
    ap.add_argument("--save_gif", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    num_classes = len(np.unique(y_train))
    N, T, F = X_test.shape
    print(f"Test set: {N} samples, classes={num_classes}")

    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

    model = load_model(device, num_classes, T, F)
    print(f"Μοντέλο: Proxy Anchor fine-tuned (78.3% Top-1)")

    rng = np.random.default_rng()
    indices = [args.index] if args.index is not None else list(rng.choice(N, args.num_samples, replace=False))

    for idx in indices:
        x_raw = X_test[idx]
        y_true = int(y_test[idx])

        x_norm = (x_raw - mean[0]) / std[0]
        xb = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0).to(device)
        xb_raw = torch.tensor(x_raw, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            try:
                logits = model(xb, xb_raw)
            except TypeError:
                logits = model(xb)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        top5 = np.argsort(probs)[::-1][:5]
        y_pred = int(top5[0])
        correct = y_pred == y_true

        print(f"\n===== Sample #{idx} =====")
        print(f"True: {y_true} | Predicted: {y_pred} | {'CORRECT' if correct else 'INCORRECT'}")

        #παράδειγμα της predicted κλάσης από το training set
        pred_examples = np.where(y_train == y_pred)[0]
        pred_pose = X_train[pred_examples[0]].reshape(T, 75, 3) if len(pred_examples) else None

        test_pose = x_raw.reshape(T, 75, 3)

        
        fig = plt.figure(figsize=(16, 6))
        gs = GridSpec(1, 3, width_ratios=[1, 1, 0.9])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])

        scat1, lines1 = setup_skeleton_ax(
            ax1, f"TEST sample #{idx}\n(True: Class {y_true})", color='#3498db')
        pred_title = f"TRAIN example of PREDICTED\nClass {y_pred} ({probs[y_pred]*100:.1f}%)"
        scat2, lines2 = setup_skeleton_ax(
            ax2, pred_title, color='#27ae60' if correct else '#e74c3c')

        #Top-5
        colors = ['#2ecc71' if c == y_true else '#e74c3c' if i == 0 else '#95a5a6'
                  for i, c in enumerate(top5)]
        ax3.barh([f"Class {c}" for c in top5][::-1], probs[top5][::-1] * 100,
                 color=colors[::-1])
        ax3.set_xlabel("Probability (%)")
        res = "CORRECT " if correct else "INCORRECT "
        ax3.set_title(f"Top-5 — {res}", fontsize=11, fontweight='bold',
                      color='#27ae60' if correct else '#c0392b')
        ax3.set_xlim(0, 100)

        def update(f):
            arts = draw_frame(test_pose[f], scat1, lines1)
            if pred_pose is not None:
                arts += draw_frame(pred_pose[f], scat2, lines2)
            return arts

        ani = animation.FuncAnimation(fig, update, frames=T, interval=60,
                                      blit=True, repeat=True)
        plt.tight_layout()
        if args.save_gif:
            ani.save(f"demo_sample_{idx}.gif", writer="pillow", fps=15)
            print(f"Αποθηκεύτηκε: demo_sample_{idx}.gif")
        plt.show()
        plt.close(fig)
        del ani

if __name__ == "__main__":
    main()
