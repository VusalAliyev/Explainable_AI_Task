# gradcampp_runner.py
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget

import segmentation_models_pytorch as smp

# PSPNet Load
def load_model(checkpoint_path):
    model = smp.PSPNet(encoder_name="resnet34", in_channels=3, classes=1)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()
    return model

# Örnek görüntüyü yükle
def load_sample_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256))
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    tensor_img = torch.tensor(rgb_img.transpose(2, 0, 1)).unsqueeze(0).float()
    return rgb_img, tensor_img

# Grad-CAM++ hesapla
def run_gradcampp(image_path, checkpoint_path, save_path="cam_output.jpg"):
    model = load_model(checkpoint_path)

    # PSPNet'te encoder'ın son layer'ı genellikle iyi çalışır
    target_layers = [model.encoder.layer4[-1]]

    rgb_img, input_tensor = load_sample_image(image_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
    cam.activations_and_grads = cam.activations_and_grads


    targets = [SemanticSegmentationTarget(0, np.random.rand(256, 256))]  # Class 0 için dummy target

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print("Grad-CAM++ çıktısı kaydedildi:", save_path)

# Örnek kullanım
if __name__ == "__main__":
    image_path = "data/Subset_I_flattened/Subset_I_00004_LAD_image-00008.jpg"  # Buraya 1 test img koy
    checkpoint_path = "checkpoints/pspnet_subset_i.pth"
    run_gradcampp(image_path, checkpoint_path)
