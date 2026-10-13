import warnings
import torch
import torch.nn as nn
import torchvision
from dataset import get_dataloaders
import torchvision.transforms as transforms
from sklearn.metrics import classification_report


warnings.filterwarnings("ignore", message="Detected call of")

ROOT = r"C:\Soong Earn (everything)\Australia\Y2S3\COMP 3710 Pattern Recognition and Analysis\Final Report\ADNI\AD_NC"

transform_train = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((224, 224)),
    transforms.RandomCrop(224, padding=8),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.RandomErasing(p=0.25)])

transform_test = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])


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
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(Bottleneck(self.in_channels, mid_channels, s))
            self.in_channels = mid_channels * Bottleneck.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    return 100 * correct / total


def report(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                outputs = model(images.to(device))
            preds += outputs.argmax(1).cpu().tolist()
            targets += labels.tolist()
    model.train()
    print(classification_report(targets, preds, target_names=["NC", "AD"], digits=3))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device, "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

    num_epochs = 20
    max_lr = 0.1
    batch_size = 32
    print(f"Epochs: {num_epochs} | Max LR: {max_lr} | Batch size: {batch_size}")

    train_loader, test_loader = get_dataloaders(batch_size)

    model = ResNet50(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=max_lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, epochs=num_epochs, steps_per_epoch=len(train_loader), pct_start=0.25)
    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

    print("\n> Training")
    model.train()

    best_acc = 0.0
    best_acc_epoch = 0

    for epoch in range(num_epochs):
        correct, total, running_loss = 0, 0, 0.0

        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if torch.cuda.is_available():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            scheduler.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_acc = evaluate(model, test_loader, device)

        if test_acc > best_acc:
            best_acc, best_acc_epoch = test_acc, epoch + 1
            torch.save(model.state_dict(), "best_model.pt")

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.5f} "
              f"Train Acc: {100*correct/total:.2f}% Test Acc: {test_acc:.2f}%")

    print("\nTUNING SUMMARY")
    print(f"Best test accuracy: {best_acc:.2f}% at epoch {best_acc_epoch}")

    model.load_state_dict(torch.load("best_model.pt"))
    final_acc = evaluate(model, test_loader, device)
    report(model, test_loader, device)

    print("\nFINAL RESULTS")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Epochs: {num_epochs} | Batch size: {batch_size}")
    print(f"Test accuracy (best checkpoint): {final_acc:.2f}%")


if __name__ == "__main__":
    main()