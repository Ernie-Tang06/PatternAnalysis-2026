import os
import sys
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, recall_score, f1_score


ROOT = r"C:\Soong Earn (everything)\Australia\Y2S3\COMP 3710 Pattern Recognition and Analysis\Final Report\ADNI\AD_NC"
EPOCHS = 30
LR = 1e-4
SEED = 0
CKPT = "ConvNeXt_for_comparison_run1.pth" # save best weights so the model can be reloaded without retraining
scaler = torch.amp.GradScaler("cuda") # supports mixed precision training


class ADNIDataset(Dataset): # returns (1-channel image tensor, label) for one sample
    def __init__(self, samples, transform):
        self.samples, self.transform = samples, transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        return self.transform(Image.open(path).convert("L")), label


def get_dataloaders(batch_size=32):
    def load(split): # list of (path, label), NC = 0, AD = 1
        return [(os.path.join(ROOT, split, c, f), int(c == "AD"))
                for c in ("NC", "AD")
                for f in sorted(os.listdir(os.path.join(ROOT, split, c))) if f.endswith(".jpeg")]

    def pid(path): # patient id from filename
        return os.path.basename(path).rsplit("_", 1)[0]

    train_all, test = load("train"), load("test")
    patients = sorted({pid(p) for p, _ in train_all}) # split by patient so no patient is in both train and val (no leakage)
    random.Random(42).shuffle(patients)
    val_ids = set(patients[:int(len(patients) * 0.2)])
    train = [s for s in train_all if pid(s[0]) not in val_ids]
    val = [s for s in train_all if pid(s[0]) in val_ids]

    plain = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    aug = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomAffine(10, (0.05, 0.05), (0.95, 1.05)),
                              transforms.ColorJitter(0.2, 0.2), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]) # augmentation for training only

    def loader(samples, tf, shuffle):
        return DataLoader(ADNIDataset(samples, tf), batch_size, shuffle=shuffle,
                          num_workers=4, pin_memory=True, persistent_workers=True)

    return loader(train, aug, True), loader(val, plain, False), loader(test, plain, False)


class LayerNorm(nn.Module): # supports both (B, H, W, C) and (B, C, H, W) layouts
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class Block(nn.Module): # ConvNeXt block: 7x7 depthwise conv, then a 2-layer MLP, plus a residual connection
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return input + x


class ConvNeXt(nn.Module):
    def __init__(self, in_chans=1, num_classes=2,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem (4x4 patchify) + 3 downsampling layers between stages
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)))

        self.stages = nn.ModuleList() # 4 stages of stacked blocks (3, 3, 9, 3)
        for i in range(4):
            self.stages.append(nn.Sequential(*[Block(dims[i]) for _ in range(depths[i])]))

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, then norm

    def forward(self, x):
        return self.head(self.forward_features(x))


def convnext(**kwargs):
    return ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)


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

    train_loader, val_loader, test_loader = get_dataloaders()
    model = convnext().cuda() # transfer the weights to the GPU
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{n_params:.2f}M params | train/val/test: {len(train_loader.dataset)} {len(val_loader.dataset)} {len(test_loader.dataset)} | lr {LR} | seed {SEED}") # sum and print the parameters, print learning rate and seed in the output
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