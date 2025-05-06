import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from PIL import Image


class TinyImageNetDataset(Dataset):
    def __init__(self, split, transform=None):
        ds = load_dataset("zh-plus/tiny-imagenet", split=split)
        self.images = ds["image"]
        self.labels = ds["label"]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label

class DataManager:
    """
    Manager for Tiny ImageNet (zh-plus/tiny-imagenet) via Hugging Face Datasets.
    """
    def __init__(self, batch_size, num_workers=2):
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform_train = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.transform_valid = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def get_loaders(self):
        train_set = TinyImageNetDataset(split="train", transform=self.transform_train)
        valid_set = TinyImageNetDataset(split="valid", transform=self.transform_valid)

        train_loader = DataLoader(train_set, batch_size=self.batch_size,
                                  shuffle=True, num_workers=self.num_workers)
        valid_loader = DataLoader(valid_set, batch_size=self.batch_size,
                                  shuffle=False, num_workers=self.num_workers)

        return train_loader, valid_loader