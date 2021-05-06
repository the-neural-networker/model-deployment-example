import io
import os 
import sys 
sys.path.append(os.path.abspath(os.path.pardir))

from PIL import Image
from torchvision import transforms 


def get_image_transforms():   
    train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225))
        ])

    test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225))
    ])

    return train_transform, test_transform

def get_train_transform():
    return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225))
        ])

def get_test_transform():
    return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225))
    ])

def transform_image(image_bytes):
    transform = get_test_transform()
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)