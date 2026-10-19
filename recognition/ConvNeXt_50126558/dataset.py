import os
import random
from collections import Counter
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

ROOT = r"C:\Soong Earn (everything)\Australia\Y2S3\COMP 3710 Pattern Recognition and Analysis\Final Report\ADNI\AD_NC"
USE_CROP = True


def crop(im):
    return im.crop((32, 8, 232, 210))


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

    idx = lambda p: int(p.rsplit("_", 1)[1].split(".")[0])

    train_all, test = load("train"), load("test")
    patients = sorted({pid(p) for p, _ in train_all})
    random.Random(42).shuffle(patients)
    val_ids = set(patients[:int(len(patients) * 0.2)])
    train = [s for s in train_all if pid(s[0]) not in val_ids]
    val = [s for s in train_all if pid(s[0]) in val_ids]

    cnt = Counter((idx(p), y) for p, y in train)
    keep = {i for i, _ in cnt if cnt[(i, 0)] >= 50 and cnt[(i, 1)] >= 50}
    train = [s for s in train if idx(s[0]) in keep]
    val = [s for s in val if idx(s[0]) in keep]
    weights = [1 / cnt[(idx(p), y)] for p, y in train]
    print("kept slice indices:", min(keep), "to", max(keep))

    pre = [transforms.Lambda(crop)] if USE_CROP else []
    plain = transforms.Compose([*pre, transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    aug = transforms.Compose([*pre, transforms.Resize((224, 224)), transforms.RandomAffine(10, (0.05, 0.05), (0.95, 1.05)),
                              transforms.ColorJitter(0.2, 0.2), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    def loader(samples, tf, shuffle, sampler=None):
        return DataLoader(ADNIDataset(samples, tf), batch_size, shuffle=shuffle and sampler is None, sampler=sampler,
                          num_workers=4, pin_memory=True, persistent_workers=True)

    return loader(train, aug, True, WeightedRandomSampler(weights, len(train))), loader(val, plain, False), loader(test, plain, False)


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    x, y = next(iter(train_loader))
    print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset), x.shape, y[:8])