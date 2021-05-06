import os 
import sys 
sys.path.append(os.path.abspath(os.path.pardir))

import torch 
from src.model import Net 
from src.cam import CamModel, create_cam
from src.utils.transform import transform_image

import cv2

def get_model(checkpoint_dir):
    model = Net()
    model.load_state_dict(torch.load(checkpoint_dir))
    model.eval()
    return model 

def get_cam_model(model):
    model = CamModel(model, "layer4")
    return model


checkpoint_dir = "./checkpoints/net-epoch=12-val_loss=0.04.pth.tar"
model = get_model(checkpoint_dir)
weights = list(list(model.children())[0].children())[-1].weight.cpu()
biases = list(list(model.children())[0].children())[-1].bias.cpu()
cam_model = get_cam_model(model)

def get_prediction(image_bytes):
    try:
        image = transform_image(image_bytes=image_bytes)
        output, feature_maps = cam_model(image)
        cam = create_cam(feature_maps, weights, biases)
        output = torch.softmax(output, 1)

        cam = cam.squeeze(0).detach().numpy()

        image = image[0].permute(1, 2, 0).detach()
        image = image * torch.tensor([[0.229, 0.224, 0.225]]) + torch.tensor([[0.485, 0.456, 0.406]])
        image = (image * 255).to(torch.uint8).numpy()
        
        cam0 = cv2.applyColorMap(cam[0], cv2.COLORMAP_JET)
        cam1 = cv2.applyColorMap(cam[1], cv2.COLORMAP_JET)
        result0 = 0.3 * cam0 + 0.5 * image
        result1 = 0.3 * cam1 + 0.5 * image
    except Exception:
        return 0, 'error'
    _, y_hat = output.max(1)
    if y_hat == 0:
        cv2.imwrite('static/cam.png', result0)
        return "Cat"
    else:
        cv2.imwrite('static/cam.png', result1)
        return "Dog"