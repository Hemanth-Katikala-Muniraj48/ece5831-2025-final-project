# train_sequence_nn.py
# ------------------------------------------------------------
# BiLSTM training on windowed pose features.
# - Runs FULL EPOCHS (no early stopping)
# - Saves best.pt and last.pt
# - Logs per-epoch metrics to CSV
# - Writes YOLO-like report + confusion matrices
# - Saves plots: train_loss.png, val_acc.png, lr.png, precision/recall/f1 per-class
# - [YOLO-PLOT] Adds 'results.png' like YOLO: loss + accuracy, with best acc annotated
# ------------------------------------------------------------

import os, json, csv
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# ----------------- CONFIG -----------------
CSV_PATH    = "ranges_dataset_grouped.csv"
RUN_DIR     = Path("runs/train_bilstm")
MODEL_BEST  = RUN_DIR / "best.pt"
MODEL_LAST  = RUN_DIR / "last.pt"
SCALER_OUT  = RUN_DIR / "scaler.joblib"
LABELS_OUT  = RUN_DIR / "labels.json"
LOG_CSV     = RUN_DIR / "results.csv"
REPORT_TXT  = RUN_DIR / "report.txt"
CM_RAW_PNG  = RUN_DIR / "confusion_matrix.png"
CM_NORM_PNG = RUN_DIR / "confusion_matrix_norm.png"
RESULTS_PNG = RUN_DIR / "results.png"   # [YOLO-PLOT] new combined figure

FEATURE_COLS = [
    "lknee","rknee","lhip","rhip","lelbow","relbow",
    "lshoulder","rshoulder","hip_y","wrist_y"
]

WINDOW_SIZE   = 60   # ~2s @ 30 fps
HOP_SIZE      = 15
BATCH_SIZE    = 64
EPOCHS        = 50
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
HIDDEN_SIZE   = 128
NUM_LAYERS    = 2
DROPOUT       = 0.3
BIDIR         = True
SEED          = 42
VAL_SPLIT     = 0.25

# LR scheduler (ReduceLROnPlateau)
SCHED_FACTOR   = 0.5
SCHED_PATIENCE = 2
GRAD_CLIP      = 5.0
# ------------------------------------------


# ----------------- Data -----------------
class SequenceDataset(Dataset):
    def __init__(self, feats_all, index_pairs):
        self.X = feats_all
        self.idxs = index_pairs  # list of (start_idx, class_idx)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        s, y = self.idxs[i]
        x = self.X[s:s+WINDOW_SIZE]  # (T,F)
        return torch.from_numpy(x).float(), torch.tensor(y).long()

def build_windows(df, lbl2i, window, hop):
    """
    Build windows within each class using the sorted original row indices.
    Assumes frames for a clip/class are contiguous in the CSV.
    """
    out = []
    for lbl in df["label"].unique():
        sub = df[df["label"] == lbl]
        idxs = sub.index.to_numpy()
        idxs.sort()
        for k in range(0, len(idxs) - window + 1, hop):
            out.append((idxs[k], lbl2i[lbl]))
    return out

def split_train_val(df, test_size=VAL_SPLIT, seed=SEED):
    X = df.index.to_numpy()
    y = df["label"].to_numpy()
    tr, va = train_test_split(X, test_size=test_size, stratify=y, random_state=seed)
    return df.loc[tr].sort_index(), df.loc[va].sort_index()


# ----------------- Model -----------------
class BiLSTMClassifier(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=2, bidirectional=True, dropout=0.3, num_classes=4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim, hidden_size=hidden, num_layers=layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if layers > 1 else 0.0
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)      # (B,T,H*)
        feat = out[:, -1, :]       # last timestep
        return self.head(feat)


# ----------------- Utils -----------------
def evaluate_full(model, loader, device, class_names):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            pred = logits.argmax(1)
            y_true += yb.cpu().tolist()
            y_pred += pred.cpu().tolist()
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = (y_true == y_pred).mean() if len(y_true) else 0.0
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    return acc, prec, rec, f1, support, cm, report

def plot_confusion_matrix(cm, class_names, normalize=False, out_path="cm.png", title="Confusion Matrix"):
    if normalize:
        with np.errstate(all='ignore'):
            cm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def current_lr(optimizer):
    for g in optimizer.param_groups:
        return g.get("lr", None)

def plot_curve(x, y, xlabel, ylabel, title, out_path):
    plt.figure(figsize=(6,4))
    plt.plot(x, y, marker='o')
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_multi_class(x, ys_dict, xlabel, ylabel, title, out_path):
    plt.figure(figsize=(7,5))
    for cls, series in ys_dict.items():
        plt.plot(x, series, marker='o', label=cls)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# [YOLO-PLOT] combined results figure like YOLO's results.png
def plot_results_like_yolo(epochs, loss_curve, acc_curve, best_epoch, best_acc, out_path):
    plt.figure(figsize=(10,4))
    # Left: Train Loss
    ax1 = plt.subplot(1,2,1)
    ax1.plot(epochs, loss_curve, marker='o')
    ax1.set_title("Train Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    # Right: Val Accuracy with best annotated
    ax2 = plt.subplot(1,2,2)
    ax2.plot(epochs, acc_curve, marker='o', label='val_acc')
    ax2.scatter([best_epoch], [best_acc], s=80, marker='*', label=f'best={best_acc:.3f} @ ep {best_epoch}')
    ax2.set_title(f"Val Accuracy (best={best_acc:.3f} @ epoch {best_epoch})")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0.0, 1.0)  # accuracy in [0,1]
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ----------------- Main -----------------
def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    missing = [c for c in FEATURE_COLS + ["label"] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    labels = sorted(df["label"].unique().tolist())
    lbl2i = {c:i for i,c in enumerate(labels)}
    print("[INFO] Classes:", labels)
    print("[INFO] Counts:", Counter(df["label"]))

    # split frames → scale on train only
    df_tr, df_va = split_train_val(df)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(df_tr[FEATURE_COLS].to_numpy())
    Xva = scaler.transform(df_va[FEATURE_COLS].to_numpy())
    joblib.dump(scaler, SCALER_OUT)
    with open(LABELS_OUT, "w") as f:
        json.dump(labels, f)
    print(f"[OK] Saved scaler -> {SCALER_OUT}")
    print(f"[OK] Saved labels -> {LABELS_OUT}")

    # Pack back into one array aligned to original df index
    feats_all = np.zeros((len(df), len(FEATURE_COLS)), dtype=np.float32)
    feats_all[df_tr.index] = Xtr.astype(np.float32)
    feats_all[df_va.index] = Xva.astype(np.float32)

    # windows
    win_tr = build_windows(df_tr, lbl2i, WINDOW_SIZE, HOP_SIZE)
    win_va = build_windows(df_va, lbl2i, WINDOW_SIZE, HOP_SIZE)
    print(f"[INFO] Train windows: {len(win_tr)} | Val windows: {len(win_va)}")

    dl_tr = DataLoader(SequenceDataset(feats_all, win_tr), batch_size=BATCH_SIZE, shuffle=True)
    dl_va = DataLoader(SequenceDataset(feats_all, win_va), batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMClassifier(
        in_dim=len(FEATURE_COLS), hidden=HIDDEN_SIZE, layers=NUM_LAYERS,
        bidirectional=BIDIR, dropout=DROPOUT, num_classes=len(labels)
    ).to(device)

    opt  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=SCHED_FACTOR, patience=SCHED_PATIENCE)
    loss = nn.CrossEntropyLoss()

    # init log csv
    with open(LOG_CSV, "w", newline="") as fp:
        w = csv.writer(fp)
        header = ["epoch","train_loss","val_acc","lr"] + \
                 [f"precision_{c}" for c in labels] + \
                 [f"recall_{c}" for c in labels] + \
                 [f"f1_{c}" for c in labels]
        w.writerow(header)

    # histories for plots
    hist_epochs, hist_loss, hist_acc, hist_lr = [], [], [], []
    hist_prec = {c: [] for c in labels}
    hist_rec  = {c: [] for c in labels}
    hist_f1   = {c: [] for c in labels}

    best_acc = 0.0
    best_ep  = 1
    for ep in range(1, EPOCHS+1):
        model.train()
        run = 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            l = loss(logits, yb)
            l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            run += l.item() * yb.size(0)
        tr_loss = run / max(1, len(dl_tr.dataset))

        # validation
        val_acc, prec, rec, f1, support, cm, report = evaluate_full(model, dl_va, device, labels)
        prev_lr = current_lr(opt)
        sch.step(val_acc)
        new_lr = current_lr(opt)

        # log line (CSV)
        with open(LOG_CSV, "a", newline="") as fp:
            w = csv.writer(fp)
            w.writerow([ep, f"{tr_loss:.6f}", f"{val_acc:.6f}", f"{new_lr if new_lr is not None else prev_lr}"] +
                       [f"{x:.6f}" for x in prec] + [f"{x:.6f}" for x in rec] + [f"{x:.6f}" for x in f1])

        # accumulate histories for plots
        hist_epochs.append(ep)
        hist_loss.append(tr_loss)
        hist_acc.append(val_acc)
        hist_lr.append(new_lr if new_lr is not None else prev_lr)
        for i, c in enumerate(labels):
            hist_prec[c].append(float(prec[i]))
            hist_rec[c].append(float(rec[i]))
            hist_f1[c].append(float(f1[i]))

        # console display
        print(f"Epoch {ep:02d}/{EPOCHS} | train_loss={tr_loss:.4f} | val_acc={val_acc:.4f} | lr={hist_lr[-1]}")
        for i, c in enumerate(labels):
            print(f"  {c:12s}  P={prec[i]:.3f}  R={rec[i]:.3f}  F1={f1[i]:.3f}  (n={support[i]})")

        # save best
        if val_acc > best_acc:
            best_acc = val_acc
            best_ep  = ep
            torch.save({
                "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
                "config": {
                    "in_dim": len(FEATURE_COLS),
                    "hidden": HIDDEN_SIZE,
                    "layers": NUM_LAYERS,
                    "bidirectional": BIDIR,
                    "dropout": DROPOUT,
                    "num_classes": len(labels),
                    "window_size": WINDOW_SIZE,
                    "feature_cols": FEATURE_COLS,
                },
                "labels": labels
            }, MODEL_BEST)
            print(f"[OK] Saved best -> {MODEL_BEST} (acc={best_acc:.4f})")

        # always save last
        torch.save({
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "config": {
                "in_dim": len(FEATURE_COLS),
                "hidden": HIDDEN_SIZE,
                "layers": NUM_LAYERS,
                "bidirectional": BIDIR,
                "dropout": DROPOUT,
                "num_classes": len(labels),
                "window_size": WINDOW_SIZE,
                "feature_cols": FEATURE_COLS,
            },
            "labels": labels
        }, MODEL_LAST)

    # ===== Final evaluation on val (for reports) =====
    model.eval()
    ckpt = torch.load(MODEL_BEST, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])

    val_acc, prec, rec, f1, support, cm, report = evaluate_full(model, dl_va, device, labels)

    # Save text report
    with open(REPORT_TXT, "w") as f:
        f.write("=== Classification Report (best model) ===\n")
        f.write(report + "\n")
        f.write(f"Val accuracy: {val_acc:.4f}\n")
        f.write(f"Best epoch: {best_ep}\n")

    # Plot and save confusion matrices
    plot_confusion_matrix(cm, labels, normalize=False, out_path=CM_RAW_PNG, title="Confusion Matrix (Counts)")
    plot_confusion_matrix(cm, labels, normalize=True,  out_path=CM_NORM_PNG, title="Confusion Matrix (Normalized)")

    # ===== Training curves (plots) =====
    plot_curve(hist_epochs, hist_loss, "Epoch", "Train Loss",
               "Training Loss", RUN_DIR / "train_loss.png")
    plot_curve(hist_epochs, hist_acc, "Epoch", "Val Accuracy",
               "Validation Accuracy", RUN_DIR / "val_acc.png")
    plot_curve(hist_epochs, hist_lr, "Epoch", "Learning Rate",
               "Learning Rate (ReduceLROnPlateau)", RUN_DIR / "lr.png")

    plot_multi_class(hist_epochs, hist_prec, "Epoch", "Precision",
                     "Per-class Precision", RUN_DIR / "precision_per_class.png")
    plot_multi_class(hist_epochs, hist_rec, "Epoch", "Recall",
                     "Per-class Recall", RUN_DIR / "recall_per_class.png")
    plot_multi_class(hist_epochs, hist_f1, "Epoch", "F1 Score",
                     "Per-class F1", RUN_DIR / "f1_per_class.png")

    # [YOLO-PLOT] Combined results figure with best acc annotation
    plot_results_like_yolo(hist_epochs, hist_loss, hist_acc, best_ep, best_acc, RESULTS_PNG)

    # Print YOLO-like summary
    print("\n==================== SUMMARY (best) ====================")
    print(f"Val Acc: {val_acc:.4f} (best={best_acc:.4f} @ epoch {best_ep})")
    print("Class              Precision   Recall   F1    Support")
    for i, c in enumerate(labels):
        print(f"{c:16s}   {prec[i]:7.3f}   {rec[i]:7.3f}  {f1[i]:7.3f}  {support[i]:7d}")
    print("Artifacts saved to:", RUN_DIR.resolve())
    print(" - best.pt / last.pt")
    print(" - results.csv + plots: train_loss.png, val_acc.png, lr.png, results.png")
    print(" - precision_per_class.png / recall_per_class.png / f1_per_class.png")
    print(" - report.txt, confusion_matrix.png, confusion_matrix_norm.png")
    print("=======================================================")

if __name__ == "__main__":
    main()
