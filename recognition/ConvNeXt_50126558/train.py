import os
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from modules import convnext
from dataset import get_dataloaders
from sklearn.metrics import classification_report, confusion_matrix

EPOCHS = 20
LR = 1e-4
SEED = 0
scaler = torch.amp.GradScaler("cuda")


def run_epoch(model, loader, loss_fn, opt=None):
    model.train(opt is not None)
    total, preds, labels = 0, [], []
    with torch.set_grad_enabled(opt is not None):
        for x, y in loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda"):
                out = model(x)
                loss = loss_fn(out, y)
            if opt:
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            total += loss.item() * len(y)
            preds.append(out.argmax(1))
            labels.append(y)
    preds, labels = torch.cat(preds), torch.cat(labels)
    return total / len(labels), (preds == labels).float().mean().item(), preds, labels


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader, test_loader = get_dataloaders()
    model = convnext().cuda()
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params | lr {LR} | seed {SEED}")
    torch.cuda.reset_peak_memory_stats()

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05},
                             {"params": no_decay, "weight_decay": 0.0}], lr=LR)
    warm = torch.optim.lr_scheduler.LinearLR(opt, 0.1, 1.0, total_iters=3)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS - 3)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], [3])

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best = 0
    for e in range(EPOCHS):
        t = time.time()
        tl, ta, _, _ = run_epoch(model, train_loader, loss_fn, opt)
        vl, va, _, _ = run_epoch(model, val_loader, loss_fn)
        sched.step()
        for k, v in zip(hist, (tl, vl, ta, va)):
            hist[k].append(v)
        if va > best:
            best = va
            torch.save(model.state_dict(), "convnext_best.pth")
        print(f"epoch {e + 1}/{EPOCHS}  train loss {tl:.4f} acc {ta:.4f}  "
              f"val loss {vl:.4f} acc {va:.4f}  time {time.time() - t:.1f}s")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, name in zip(ax, ("loss", "acc")):
        a.plot(hist[f"train_{name}"], label="train")
        a.plot(hist[f"val_{name}"], label="val")
        a.set_xlabel("epoch")
        a.set_title(name)
        a.legend()
    plt.tight_layout()
    plt.savefig("training_curves.png")

    model.load_state_dict(torch.load("convnext_best.pth"))
    test_loss, test_acc, p, l = run_epoch(model, test_loader, loss_fn)
    print(f"best val acc {best:.4f}")
    print(f"test loss {test_loss:.4f}  test acc {test_acc:.4f}")
    print(classification_report(l.cpu(), p.cpu(), target_names=["NC", "AD"], digits=4))
    print(confusion_matrix(l.cpu(), p.cpu()))
    print(f"peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")