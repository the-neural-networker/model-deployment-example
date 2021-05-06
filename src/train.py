import os 
import sys 
sys.path.append(os.path.abspath(os.path.pardir))
from argparse import ArgumentParser, Namespace

import torch 
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from src.dataset import CatsVsDogsDataModule 
from src.model import Net 
from src.utils.transform import get_image_transforms


def main() -> None: 
    args = get_args() 

    train_transform, test_transform = get_image_transforms()
    
    dm = CatsVsDogsDataModule(
        data_dir=args.data_dir,
        train_transform=train_transform,
        test_transform=test_transform,
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )

    model = Net(n_classes=args.n_classes, learning_rate=args.learning_rate)

    early_stopping = EarlyStopping('val_loss', patience=5)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="../checkpoints/",
        filename="net-{epoch:02d}-{val_loss:.2f}",
        save_top_k=-1
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(gpus=args.gpus,
                        max_epochs=args.max_epochs, 
                        limit_train_batches=args.limit_train_batches, 
                        limit_val_batches=args.limit_val_batches,
                        limit_test_batches=args.limit_test_batches,
                        callbacks=[early_stopping, checkpoint_callback, lr_monitor]
                    )
                    
    trainer.fit(model, dm)
    trainer.test(datamodule=dm)


def get_args():
    parser = ArgumentParser() 
    parser.add_argument("--data_dir", default="../dataset/", type=str)
    parser.add_argument("--batch_size", default=64, type=int) 
    parser.add_argument("--num_workers", default=4, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Net.add_model_specific_args(parser)
    args = parser.parse_args() 
    return args 


if __name__ == "__main__":
    main()