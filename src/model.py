import os 
import sys

sys.path.append(os.path.abspath(os.path.pardir))
from argparse import ArgumentParser

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torchvision.models import resnet18, resnet50

import pytorch_lightning as pl 

from src.dataset import CatsVsDogsDataModule

class Net(pl.LightningModule):

    def __init__(self, net="resnet18", n_classes=2, learning_rate=1e-3):
        super(Net, self).__init__()
        self.save_hyperparameters()
        if net == "resnet18":
            self.model = resnet18(pretrained=True)
        elif net == "resnet50":
            self.model = resnet50(pretrained=True) 
        else:
            raise NotImplementedError("Net not implemented.")
        
        in_features = list(self.model.children())[-1].in_features
        self.model.fc = nn.Linear(in_features=in_features, out_features=self.hparams.n_classes)

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        X, y = batch 
        y_hat = self.model(X) 
        loss = F.cross_entropy(y_hat, y)         
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss 

    def validation_step(self, batch, batch_idx):
        X, y = batch 
        y_hat = self.model(X) 
        loss = F.cross_entropy(y_hat, y) 
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        X, y = batch 
        y_hat = self.model(X) 
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--n_classes', type=int, default=2)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        return parser


if __name__ == "__main__":
    data_dir = "../dataset/"
    dm = CatsVsDogsDataModule(data_dir=data_dir)
    dm.setup()
    model = Net(n_classes=2)
    print(model)

    for X, y in dm.train_dataloader():
        out = model(X)
        print(out, y)
        print(out.shape, y.shape)
        print(F.nll_loss(out, y))
        break