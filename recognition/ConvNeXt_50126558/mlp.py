import os
import sys
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from ConvNeXt_for_comparison import get_dataloaders
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, recall_score, f1_score


EPOCHS = 30
LR = 1e-4
SEED = 0
CKPT = "mlp_run1.pth" # save best weights so the model can be reloaded without retraining
scaler = torch.amp.GradScaler("cuda") # supports mixed precision training


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(224 * 224, 553), nn.ReLU(), nn.Dropout(0.3), # 553 to match the parameters for ConvNeXt comparison
            nn.Linear(553, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 2),)

    def forward(self, x):
        return self.net(x)


class Tee: # log all printed output to file and console
    def __init__(self, path):
        self.terminal, self.file = sys.stdout, open(path, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg)
        self.file.write(msg)
        self.file.flush()

    def flush(self):
        self.terminal.flush()
        self.file.flush()


def run_epoch(model, loader, loss_fn, opt=None): 
    model.train(opt is not None)
    total, preds, labels, probs = 0, [], [], []
    with torch.set_grad_enabled(opt is not None):
        for x, y in loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True) # start CPU to GPU copy and carry on without waiting
            with torch.autocast("cuda"):
                out = model(x)
                loss = loss_fn(out, y)
            if opt: # control if optimizer is needed (needed in training, not needed in validation)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            total += loss.item() * len(y)
            preds.append(out.argmax(1)) # returns the index of the largest value
            labels.append(y)
            probs.append(out.float().softmax(1)[:, 1])
    preds, labels, probs = torch.cat(preds), torch.cat(labels), torch.cat(probs)
    return total / len(labels), (preds == labels).float().mean().item(), preds, labels, probs


if __name__ == "__main__": # setup: run from script folder, log output, fix seeds
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.stdout = Tee(CKPT.replace(".pth", ".log"))
    sys.stderr = sys.stdout # sends errors and warnings into the log
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True # remain for fair comparison

    train_loader, val_loader, test_loader = get_dataloaders() # same data preprocessing from ConvNeXt for comparison
    model = MLP().cuda() # transfer the weights to the GPU
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{n_params:.2f}M params | train/val/test: {len(train_loader.dataset)} {len(val_loader.dataset)} {len(test_loader.dataset)} | lr {LR} | seed {SEED}") # sum and print the parameters, print leraning rate and seed in the output
    torch.cuda.reset_peak_memory_stats() # resets the peak memory status

    loss_fn = nn.CrossEntropyLoss() # measures how wrong the model's prediction is
    opt = torch.optim.Adam(model.parameters(), LR) # Adam optimizer to update the model's weight

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []} # append after each epoch
    best = 0
    for e in range(EPOCHS):
        tl, ta, _, _, _ = run_epoch(model, train_loader, loss_fn, opt) # training with optimizer
        vl, va, _, _, _ = run_epoch(model, val_loader, loss_fn) # validation without optimizer
        for k, v in zip(hist, (tl, vl, ta, va)): # replace 4 append lines
            hist[k].append(v)
        if va > best: # save the best validation accuracy and overwrite the CKPT file
            best = va
            torch.save(model.state_dict(), CKPT)
        print(f"epoch {e + 1}/{EPOCHS}  train loss {tl:.4f} acc {ta:.4f}  val loss {vl:.4f} acc {va:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4)) # plot the training and loss curve
    for a, name in zip(ax, ("loss", "acc")):
        a.plot(hist[f"train_{name}"], label="train")
        a.plot(hist[f"val_{name}"], label="val")
        a.set_xlabel("epoch")
        a.set_title(name)
        a.legend()
    plt.tight_layout()
    plt.savefig(CKPT.replace(".pth", "_curves.png"))

    print(f"\nsaved best checkpoint: {CKPT}, exists: {os.path.exists(CKPT)}")
    model.load_state_dict(torch.load(CKPT))
    test_loss, test_acc, p, l, pr = run_epoch(model, test_loader, loss_fn) # run one evaluation pass on the test set
    auroc = roc_auc_score(l.cpu(), pr.cpu()) # measures how well the model ranks AD images above NC using probabilities
    vram = torch.cuda.max_memory_allocated() / 1e9 # peak memory converted to GB

    print("\n===== FINAL RESULTS =====")
    print(f"best val acc {best:.4f}")
    print(f"test loss {test_loss:.4f}  test acc {test_acc:.4f}  test AUROC {auroc:.4f}")
    print(classification_report(l.cpu(), p.cpu(), target_names=["NC", "AD"], digits=4))
    print(confusion_matrix(l.cpu(), p.cpu()))
    print(f"params {n_params:.2f}M  peak VRAM {vram:.2f} GB")

    res = {"run": CKPT, "best_val_acc": round(best, 4), "test_acc": round(test_acc, 4),
           "test_auroc": round(auroc, 4),
           "ad_recall": round(recall_score(l.cpu(), p.cpu()), 4),
           "macro_f1": round(f1_score(l.cpu(), p.cpu(), average="macro"), 4),
           "params_M": round(n_params, 2), "peak_vram_gb": round(vram, 2)}
    json.dump(res, open(CKPT.replace(".pth", "_results.json"), "w"), indent=1) # save headline metrics to JSON for comparing runs