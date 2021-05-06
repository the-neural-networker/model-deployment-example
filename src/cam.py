import os 
import sys 
sys.path.append(os.path.abspath(os.path.pardir))
from argparse import ArgumentParser, Namespace
from tqdm import tqdm 

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms 

import pytorch_lightning as pl

from src.dataset import CatsVsDogsDataModule 
from src.model import Net 
from src.utils.transform import get_image_transforms

import cv2

def main():
    args = get_args() 

    train_transform, test_transform = get_image_transforms()

    dm = CatsVsDogsDataModule(
            data_dir=args.data_dir,
            train_transform=train_transform,
            test_transform=test_transform,
            test_shuffle=True,
            batch_size=args.batch_size, 
            num_workers=args.num_workers
        )
    dm.setup()

    model = Net.load_from_checkpoint(args.checkpoint_dir)
    weights = list(list(model.children())[0].children())[-1].weight.cpu()
    biases = list(list(model.children())[0].children())[-1].bias.cpu()
    model = CamModel(model, "layer4")
    model.eval()

    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(dm.test_dataloader()):
            out, feature_maps = model(X)
            _, y_hat = out.max(1)
            cams = create_cam(feature_maps, weights, biases)
            cams = cams.detach().numpy()

            i = 0
            for cam, image in zip(cams, X):

                image = image.permute(1, 2, 0).detach()
                image = image * torch.tensor([[0.229, 0.224, 0.225]]) + torch.tensor([[0.485, 0.456, 0.406]])
                image = (image * 255).to(torch.uint8).numpy()

                
                cam0 = cv2.applyColorMap(cam[0], cv2.COLORMAP_JET)
                cam1 = cv2.applyColorMap(cam[1], cv2.COLORMAP_JET)
                result0 = 0.3 * cam0 + 0.5 * image
                result1 = 0.3 * cam1 + 0.5 * image

                if not os.path.exists(args.cam_path):
                    os.makedirs(args.cam_path)
                if y_hat == 0:
                    cv2.imwrite(args.cam_path + f"batch_{batch_idx}_index_{i}_cat.png", result0)
                else:
                    cv2.imwrite(args.cam_path + f"batch_{batch_idx}_index_{i}_dog.png", result1)
                    
                i += 1
            
            if batch_idx == args.num_batches - 1:
                break

class CamModel(nn.Module):
    def __init__(self, model, output_layer="layer4"):
        super().__init__()
        self.output_layer = output_layer
        self.model = model
        self.children_list = []
        for n,c in self.model.model.named_children():
            self.children_list.append(c)
            if n == self.output_layer:
                break

        self.net = nn.Sequential(*self.children_list)
        
    def forward(self,x):
        out = self.model(x)
        x = self.net(x)
        return out, x

def create_cam(feature_maps, weights, biases):
    batch_size, n_channels, height, width = feature_maps.shape 
    n_classes = weights.shape[0]
    maps = torch.matmul(weights, feature_maps.view(batch_size, n_channels, height * width))
    maps = maps.view(batch_size, n_classes, height, width)
    maps = maps + biases.view(1, -1, 1, 1)
    maps = maps - maps.min()
    maps = maps / maps.max() 
    maps = (maps * 255)
    maps = F.interpolate(maps, size=(224, 224), mode="bilinear", align_corners=False)
    return maps.to(torch.uint8)

def get_args():
    parser = ArgumentParser() 
    parser.add_argument("--data_dir", default="../dataset/", type=str)
    parser.add_argument("--batch_size", default=1, type=int) 
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--checkpoint_dir", default="../checkpoints/net-epoch=12-val_loss=0.04.ckpt", type=str)
    parser.add_argument("--cam_path", default="../class_activation_maps/", type=str)
    parser.add_argument("--num_batches", default=5, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args() 
    return args


if __name__ == "__main__":
    main()