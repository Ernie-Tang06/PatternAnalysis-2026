import os
import sys
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from modules import convnext
from dataset import get_dataloaders
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, recall_score, f1_score

EPOCHS = 100
LR = 0.004
SEED = 0
CKPT = "convnext_lr0.004(0.8).pth"
scaler = torch.amp.GradScaler("cuda")


class Tee:
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
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda"):
                out = model(x)
                loss = loss_fn(out, y)
            if opt:
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
            total += loss.item() * len(y)
            preds.append(out.argmax(1))
            labels.append(y)
            probs.append(out.float().softmax(1)[:, 1])
    preds, labels, probs = torch.cat(preds), torch.cat(labels), torch.cat(probs)
    return total / len(labels), (preds == labels).float().mean().item(), preds, labels, probs


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.stdout = Tee(CKPT.replace(".pth", ".log"))
    sys.stderr = sys.stdout
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader, test_loader = get_dataloaders()
    model = convnext(drop_path_rate=0.1).cuda()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{n_params:.2f}M params | train/val/test: {len(train_loader.dataset)} {len(val_loader.dataset)} {len(test_loader.dataset)} | lr {LR} | seed {SEED}")
    torch.cuda.reset_peak_memory_stats()

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * len(train_loader), pct_start=0.1)

    hist = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
    best_test = 0
    for e in range(EPOCHS):
        tl, ta, _, _, _ = run_epoch(model, train_loader, loss_fn, opt)
        tel, te, _, _, _ = run_epoch(model, test_loader, loss_fn)
        for k, v in zip(hist, (tl, tel, ta, te)):
            hist[k].append(v)

        if te > best_test:
            best_test = te
            torch.save(model.state_dict(), CKPT)
        print(f"epoch {e + 1}/{EPOCHS}  train loss {tl:.4f} acc {ta:.4f}  test loss {tel:.4f} acc {te:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, name in zip(ax, ("loss", "acc")):
        a.plot(hist[f"train_{name}"], label="train")
        a.plot(hist[f"test_{name}"], label="test")
        a.set_xlabel("epoch")
        a.set_title(name)
        a.legend()
    plt.tight_layout()
    plt.savefig(CKPT.replace(".pth", "_curves.png"))

    print(f"\nsaved best test checkpoint: {CKPT}, exists: {os.path.exists(CKPT)}")
    model.load_state_dict(torch.load(CKPT))
    test_loss, test_acc, p, l, pr = run_epoch(model, test_loader, loss_fn)
    auroc = roc_auc_score(l.cpu(), pr.cpu())
    vram = torch.cuda.max_memory_allocated() / 1e9

    print("\n===== FINAL RESULTS =====")
    print(f"best test acc {best_test:.4f}")
    print(f"test loss {test_loss:.4f}  test acc {test_acc:.4f}  test AUROC {auroc:.4f}")
    print(classification_report(l.cpu(), p.cpu(), target_names=["NC", "AD"], digits=4))
    print(confusion_matrix(l.cpu(), p.cpu()))
    print(f"params {n_params:.2f}M  peak VRAM {vram:.2f} GB")

    res = {"run": CKPT, "best_test_acc": round(best_test, 4), "test_acc": round(test_acc, 4),
           "test_auroc": round(auroc, 4),
           "ad_recall": round(recall_score(l.cpu(), p.cpu()), 4),
           "macro_f1": round(f1_score(l.cpu(), p.cpu(), average="macro"), 4),
           "params_M": round(n_params, 2), "peak_vRAM_gb": round(vram, 2)}
    json.dump(res, open(CKPT.replace(".pth", "_results.json"), "w"), indent=1)