import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import json
import os
from pycocotools import mask as maskUtils

from models.cotr_models import MiniUNet, PlaqueDataset

# Cihaz ayarı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dönüşümler ve veri seti
transform = T.Compose([T.ToTensor()])
dataset = PlaqueDataset(
    image_dir="data/Subset_I",
    annotation_file="annotations/Subset_I.json",
    transform=transform
)

# Model yükleniyor
model = MiniUNet().to(device)
model.load_state_dict(torch.load("trained_cotr_subsetI.pth", map_location=device))
model.eval()

# Görselleştirilecek örnek indeksler
indices_to_visualize = [0, 1, 2]

for idx in indices_to_visualize:
    image, mask = dataset[idx]
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        pred_mask = (pred_mask > 0.5).astype(np.uint8)

    image_np = image.squeeze().cpu().permute(1, 2, 0).numpy()
    gt_mask = mask.squeeze().cpu().numpy()

    # Dice skoru hesaplama
    intersection = (pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()
    dice_score = (2 * intersection + 1e-6) / (union + 1e-6)

    # Görselleştirme
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    axes[0].imshow(image_np)
    axes[0].set_title("Image")
    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title("Ground Truth")
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title(f"Prediction\nDice: {dice_score:.4f}")

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"visualize_sample_{idx}.png")
    plt.show()
