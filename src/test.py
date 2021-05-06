import os 
import sys 
sys.path.append(os.path.abspath(os.path.pardir))
from argparse import ArgumentParser, Namespace
from tqdm import tqdm 

import torch 
from torchvision import transforms 

import pytorch_lightning as pl 
from torchmetrics.functional import accuracy, precision, recall, auroc 

from src.dataset import CatsVsDogsDataModule 
from src.model import Net 
from src.utils.transform import get_image_transforms


def main():
    args = get_args() 

    train_transform, test_transform = get_image_transforms()
    
    dm = CatsVsDogsDataModule(
        data_dir=args.data_dir,
        train_transform=train_transform,
        test_transform=test_transform,
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )
    dm.setup()

    model = Net.load_from_checkpoint(args.checkpoint_dir)
    model.to(device="cuda")
    model.eval()

    y_hat_proba_all = torch.zeros((0,))
    y_hat_all = torch.zeros((0,)).long()
    y_all = torch.zeros((0,)).long()

    with torch.no_grad():
        for X, y in tqdm(dm.test_dataloader()):
            X = X.cuda()
            y = y.cuda()
            y_hat = model(X)
            y_hat_soft = torch.softmax(y_hat, dim=1)
            _, y_hat = torch.max(y_hat_soft, dim=1)            
            y_hat_proba_all = torch.cat([y_hat_proba_all, y_hat_soft[:, 1].cpu()])
            y_hat_all = torch.cat([y_hat_all, y_hat.cpu()])
            y_all = torch.cat([y_all, y.cpu()])
            
    print("="*50)
    print("Accuracy: ", float(accuracy(y_hat_all, y_all)))
    print("Precision: ", float(precision(y_hat_proba_all, y_all)))
    print("Recall: ", float(recall(y_hat_proba_all, y_all)))
    print("AUROC: ", float(auroc(y_hat_proba_all, y_all, pos_label=1)))
    print("="*50)

def get_args():
    parser = ArgumentParser() 
    parser.add_argument("--data_dir", default="../dataset/", type=str)
    parser.add_argument("--batch_size", default=64, type=int) 
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--checkpoint_dir", default="../checkpoints/net-epoch=12-val_loss=0.04.ckpt", type=str)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args() 
    return args


if __name__ == "__main__":
    main()