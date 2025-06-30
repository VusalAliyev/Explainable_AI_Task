import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

# MiniUNet (CoTr)
class MiniUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Taylor yöntemi ve görsel karşılaştırma
def run_taylor(model, image_path, mask_path, checkpoint_path, save_path="taylor_compare.jpg"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Görsel yükle
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Görüntü bulunamadı: {image_path}")
    img = cv2.resize(img, (512, 512))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img_tensor = torch.tensor(img_rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    img_tensor.requires_grad = True

    # Ground Truth maskeyi yükle
    gt_mask = Image.open(mask_path).convert("L").resize((512, 512))
    gt_mask = np.array(gt_mask) / 255.0

    # Model
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    output = model(img_tensor)
    score = torch.sigmoid(output).mean()
    score.backward()

    # Saliency = input * |∂output/∂input|
    saliency = img_tensor.grad.abs() * img_tensor
    saliency = saliency.sum(dim=1).squeeze().detach().cpu().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-6)

    # Görselleştirme
    heatmap = show_cam_on_image(img_rgb, saliency, use_rgb=True)

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img_rgb)
    axs[0].set_title("Original")
    axs[1].imshow(gt_mask, cmap='gray')
    axs[1].set_title("Ground Truth")
    axs[2].imshow(heatmap)
    axs[2].set_title("Taylor Map")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Görsel karşılaştırma kaydedildi: {save_path}")

# Kullanım
if __name__ == "__main__":
    image_path = "data/Subset_II_flattened/Subset_II_00004_LAD_image-00008.jpg"
    mask_path = "masks/Subset_II/Subset_II_00004_LAD_image-00008.png"
    checkpoint_path = "checkpoints/trained_cotr_subsetII.pth"
    model = MiniUNet()
    run_taylor(model, image_path, mask_path, checkpoint_path, save_path="taylor_cotr_compare.jpg")
