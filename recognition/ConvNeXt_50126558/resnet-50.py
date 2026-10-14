import os
import time
import warnings
import torch
import torch.nn as nn
from dataset import get_dataloaders
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore", message="Detected call of")

EPOCHS = 30
MAX_LR = 0.1
SEED = 0
CKPT = "resnet_run3.pth"
scaler = torch.amp.GradScaler("cuda")


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, mid_channels, stride=1):
        super().__init__()
        out_channels = mid_channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class ResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self.make_layer(64, 3, stride=1)
        self.layer2 = self.make_layer(128, 4, stride=2)
        self.layer3 = self.make_layer(256, 6, stride=2)
        self.layer4 = self.make_layer(512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def make_layer(self, mid_channels, blocks, stride):
        layers = []
        for s in [stride] + [1] * (blocks - 1):
            layers.append(Bottleneck(self.in_channels, mid_channels, s))
            self.in_channels = mid_channels * Bottleneck.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.avgpool(x), 1))


def run_epoch(model, loader, loss_fn, opt=None, sched=None):
    model.train(opt is not None)
    total, preds, labels = 0, [], []
    with torch.set_grad_enabled(opt is not None):
        for x, y in loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda"):
                out = model(x)
                loss = loss_fn(out, y)
            if opt:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
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
    model = ResNet50().cuda()
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params | lr {MAX_LR} | seed {SEED}")
    torch.cuda.reset_peak_memory_stats()

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.SGD(model.parameters(), lr=MAX_LR, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=MAX_LR, epochs=EPOCHS, steps_per_epoch=len(train_loader), pct_start=0.25)

    best = 0
    for e in range(EPOCHS):
        t = time.time()
        tl, ta, _, _ = run_epoch(model, train_loader, loss_fn, opt, sched)
        vl, va, _, _ = run_epoch(model, val_loader, loss_fn)
        if va > best:
            best = va
            torch.save(model.state_dict(), "CKPT")
        print(f"epoch {e + 1}/{EPOCHS}  train loss {tl:.4f} acc {ta:.4f}  "
              f"val loss {vl:.4f} acc {va:.4f}  time {time.time() - t:.1f}s")

    model.load_state_dict(torch.load((CKPT)))
    test_loss, test_acc, p, l = run_epoch(model, test_loader, loss_fn)
    print(f"best val acc {best:.4f}")
    print(f"test loss {test_loss:.4f}  test acc {test_acc:.4f}")
    print(classification_report(l.cpu(), p.cpu(), target_names=["NC", "AD"], digits=4))
    print(confusion_matrix(l.cpu(), p.cpu()))
    print(f"peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"saved best checkpoint: {CKPT}, exists: {os.path.exists(CKPT)}")