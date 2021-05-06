import os 
import sys 
sys.path.append(os.path.abspath(os.path.pardir))

from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import datasets, transforms 

import pytorch_lightning as pl
from torchvision.transforms.transforms import CenterCrop

from src.utils.transform import get_image_transforms


class CatsVsDogsDataModule(pl.LightningDataModule):

    def __init__(self, data_dir, train_transform=None, test_transform=None, test_shuffle=False, batch_size=32, num_workers=4):
        super(CatsVsDogsDataModule, self).__init__()
        self.data_dir = data_dir 

        if train_transform is None: 
            self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225))
            ])
        else:
            self.train_transform = train_transform 

        if test_transform is None: 
            self.test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225))
            ])
        else:
            self.test_transform = test_transform 

        self.batch_size = batch_size 
        self.num_workers = num_workers
        self.test_shuffle = test_shuffle

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.dataset = datasets.ImageFolder(os.path.join(self.data_dir, "training_set"), transform=self.train_transform)
            self.train_set, self.val_set = random_split(self.dataset, (int(0.8 * len(self.dataset)), len(self.dataset) - int(0.8 * len(self.dataset))))
        if stage == "test" or stage is None:
            self.test_set = datasets.ImageFolder(os.path.join(self.data_dir, "training_set"), transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=self.test_shuffle, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=self.test_shuffle, num_workers=self.num_workers, pin_memory=True)
        

if __name__ == "__main__":
    data_dir = "../dataset/"
    train_transform, test_transform = get_image_transforms()
    dm = CatsVsDogsDataModule(data_dir=data_dir)
    dm.setup()

    for X, y in dm.train_dataloader():
        print(X, y)
        print(X.shape, y.shape)
        break
