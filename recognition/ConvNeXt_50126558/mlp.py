import torch, torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix, balanced_accuracy_score, matthews_corrcoef, roc_curve

root = r"C:\Soong Earn (everything)\Australia\Y2S3\COMP 3710 Pattern Recognition and Analysis\Final Report\ADNI\AD_NC"
tf = transforms.Compose([transforms.Grayscale(1), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
full = datasets.ImageFolder(root + r"\train", tf)
test_ds = datasets.ImageFolder(root + r"\test", tf)
groups = [s[0].replace("\\", "/").split("/")[-1].split("_")[0] for s in full.samples]
tr_idx, va_idx = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(range(len(full)), groups=groups))
train_dl = DataLoader(Subset(full, tr_idx), 32, shuffle=True)
val_dl = DataLoader(Subset(full, va_idx), 32)
test_dl = DataLoader(test_ds, 32)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(224 * 224, 553), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(553, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)

dev = "cuda"
model = MLP().to(dev)
print("params:", sum(p.numel() for p in model.parameters()), "train/val/test:", len(tr_idx), len(va_idx), len(test_ds))
opt = torch.optim.Adam(model.parameters(), 1e-4)
loss_fn = nn.CrossEntropyLoss()

def evaluate(dl):
    model.eval()
    loss, ys, ps = 0, [], []
    with torch.no_grad():
        for x, y in dl:
            out = model(x.to(dev))
            loss += loss_fn(out, y.to(dev)).item() * len(y)
            ps.append(out.softmax(1)[:, 1].cpu())
            ys.append(y)
    y, p = torch.cat(ys), torch.cat(ps)
    return loss / len(y), ((p > 0.5) == y).float().mean().item(), y, p

tr_loss, va_loss, va_acc, best = [], [], [], 0
for ep in range(30):
    model.train()
    total = 0
    for x, y in train_dl:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        total += loss.item() * len(y)
    vl, va, _, _ = evaluate(val_dl)
    tr_loss.append(total / len(tr_idx)); va_loss.append(vl); va_acc.append(va)
    if va > best:
        best = va
        torch.save(model.state_dict(), "mlp.pth")
    print(f"epoch {ep + 1}: train {tr_loss[-1]:.4f} val {vl:.4f} val acc {va:.4f}")

model.load_state_dict(torch.load("mlp.pth"))
_, acc, y, p = evaluate(test_dl)
y, p = y.numpy(), p.numpy()
pred = (p > 0.5).astype(int)
print(classification_report(y, pred, target_names=["AD", "NC"], digits=4))
print("confusion matrix (rows true AD, NC; cols pred AD, NC):\n", confusion_matrix(y, pred))
print(f"test acc {acc:.4f} balanced acc {balanced_accuracy_score(y, pred):.4f} MCC {matthews_corrcoef(y, pred):.4f}")
print(f"AUROC {roc_auc_score(y, p):.4f} AUPRC (AD) {average_precision_score(y == 0, 1 - p):.4f}")
print(f"best val {best:.4f} peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(tr_loss, label="train"); ax[0].plot(va_loss, label="val"); ax[0].set_title("loss"); ax[0].legend()
ax[1].plot(va_acc); ax[1].set_title("val acc")
fpr, tpr, _ = roc_curve(y, p)
ax[2].plot(fpr, tpr); ax[2].plot([0, 1], [0, 1], "--"); ax[2].set_title("ROC")
plt.savefig("mlp_curves.png")