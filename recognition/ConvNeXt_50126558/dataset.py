import os
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

    norm = T.Normalize((0.5,), (0.5,))
    plain = T.Compose([T.Resize((224, 224)), T.ToTensor(), norm])
    aug = T.Compose([T.Resize((224, 224)), T.RandomAffine(10, (0.05, 0.05), (0.95, 1.05)),
                     T.ColorJitter(0.2, 0.2), T.ToTensor(), norm])

    def loader(samples, tf, shuffle):
        return DataLoader(ADNIDataset(samples, tf), batch_size, shuffle=shuffle,
                          num_workers=2, pin_memory=True, persistent_workers=True)

    return loader(load("train"), aug, True), loader(load("test"), plain, False)


if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders()
    x, y = next(iter(train_loader))
    print(len(train_loader.dataset), len(test_loader.dataset), x.shape, y[:8])