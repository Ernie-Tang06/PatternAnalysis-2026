import os
import random
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

ROOT = r"C:\Soong Earn (everything)\Australia\Y2S3\COMP 3710 Pattern Recognition and Analysis\Final Report\ADNI\AD_NC"


class ADNIDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples, self.transform = samples, transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        return self.transform(Image.open(path).convert("L")), label


def get_dataloaders(batch_size=32):
    def load(split):
        return [(os.path.join(ROOT, split, c, f), int(c == "AD"))
                for c in ("NC", "AD")
                for f in sorted(os.listdir(os.path.join(ROOT, split, c))) if f.endswith(".jpeg")]

    def pid(path):
        return os.path.basename(path).rsplit("_", 1)[0]

    train_all, test = load("train"), load("test")
    patients = sorted({pid(p) for p, _ in train_all})
    random.Random(42).shuffle(patients)
    val_ids = set(patients[:int(len(patients) * 0.2)])
    train = [s for s in train_all if pid(s[0]) not in val_ids]
    val = [s for s in train_all if pid(s[0]) in val_ids]

    plain = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    aug = T.Compose([T.Resize((224, 224)), T.RandomAffine(10, (0.05, 0.05), (0.95, 1.05)),
                     T.ColorJitter(0.2, 0.2), T.ToTensor()])

    def loader(samples, tf, shuffle):
        return DataLoader(ADNIDataset(samples, tf), batch_size, shuffle=shuffle,
                          num_workers=2, pin_memory=True, persistent_workers=True)

    return loader(train, aug, True), loader(val, plain, False), loader(test, plain, False)

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    x, y = next(iter(train_loader))
    print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset), x.shape, y[:8])
    
import sys, torch
print(sys.executable, torch.cuda.is_available())