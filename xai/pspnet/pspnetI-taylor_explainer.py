import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import segmentation_models_pytorch as smp

# Ayarlar
IMAGE_DIR = "data/Subset_I_flattened"
MASK_DIR = "masks/Subset_I"
MODEL_PATH = "checkpoints/pspnet_subset_i.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256

# Dataset
class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.imgs = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_file = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_file)
        mask_path = os.path.join(self.mask_dir, img_file.replace(".jpg", ".png"))

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img_tensor = torch.tensor(np.transpose(img_rgb, (2, 0, 1)), dtype=torch.float32)

        mask = (mask > 0).astype(np.float32)
        mask_tensor = torch.tensor(mask).unsqueeze(0)

        return img_tensor, mask_tensor, img_rgb

# Taylor XAI
def taylor_explainer():
    os.makedirs("taylor_results", exist_ok=True)

    model = smp.PSPNet(encoder_name="resnet34", in_channels=3, classes=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    dataset = SegDataset(IMAGE_DIR, MASK_DIR)

    for i in range(5):
        img_tensor, mask_tensor, img_rgb = dataset[i]
        input_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        input_tensor.requires_grad = True

        output = model(input_tensor)
        pred = torch.sigmoid(output)
        pred_score = pred.mean()  # toplam skoru minimize/maximize eden bölgeler

        model.zero_grad()
        pred_score.backward()

        gradients = input_tensor.grad.data.squeeze().cpu().numpy()
        input_np = input_tensor.squeeze().detach().cpu().numpy()

        # Taylor heatmap: gradient × input
        taylor_map = gradients * input_np
        taylor_map = np.abs(taylor_map).mean(axis=0)  # kanal bazında ortalama
        taylor_map = (taylor_map - taylor_map.min()) / (taylor_map.max() - taylor_map.min() + 1e-6)

        # Görselleştir
        plt.figure(figsize=(10, 3))

        plt.subplot(1, 3, 1)
        plt.imshow(img_rgb)
        plt.title("Original")

        plt.subplot(1, 3, 2)
        plt.imshow(mask_tensor.squeeze(), cmap="gray")
        plt.title("Ground Truth")

        plt.subplot(1, 3, 3)
        plt.imshow(img_rgb)
        plt.imshow(taylor_map, cmap="hot", alpha=0.5)
        plt.title("Taylor Heatmap")

        plt.tight_layout()
        plt.savefig(f"taylor_results/taylor_{i+1}.png")
        plt.close()

    print("Taylor sonuçları 'taylor_results/' klasörüne kaydedildi.")

if __name__ == "__main__":
    taylor_explainer()
