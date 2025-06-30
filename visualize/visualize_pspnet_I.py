import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import segmentation_models_pytorch as smp

# Parametreler
IMAGE_DIR = "data/Subset_I_flattened"
MASK_DIR = "masks/Subset_I"
MODEL_PATH = "pspnet_subset_i.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256

# Dataset Sınıfı
class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.imgs = []
        self.masks = []

        for mask_file in sorted(os.listdir(mask_dir)):
            if not mask_file.endswith(".png"):
                continue
            img_file = mask_file.replace(".png", ".jpg")
            img_path = os.path.join(img_dir, img_file)
            mask_path = os.path.join(mask_dir, mask_file)

            if os.path.exists(img_path) and os.path.exists(mask_path):
                self.imgs.append(img_file)
                self.masks.append(mask_file)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)

        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        mask = (mask > 0).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(img), torch.tensor(mask)

# Dice Skoru
def dice_score(pred, mask):
    pred = (pred > 0.5).float()
    smooth = 1e-6
    intersection = (pred * mask).sum()
    return (2. * intersection + smooth) / (pred.sum() + mask.sum() + smooth)

# Görselleştirme
def visualize_predictions():
    os.makedirs("results", exist_ok=True)

    # Model yükle
    model = smp.PSPNet(encoder_name="resnet34", in_channels=3, classes=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    dataset = SegDataset(IMAGE_DIR, MASK_DIR)

    for i in range(7):  # İlk 7 örnek
        img, mask = dataset[i]
        img_tensor = img.unsqueeze(0).to(DEVICE)
        mask = mask.to(DEVICE)

        with torch.no_grad():
            pred = model(img_tensor)
        pred = torch.sigmoid(pred)

        dice = dice_score(pred, mask).item()

        # Görseller
        img_np = img_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
        mask_np = mask.squeeze().cpu().numpy()
        pred_np = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

        # Plot
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(img_np)
        axs[0].set_title("Input Image")
        axs[1].imshow(mask_np, cmap='gray')
        axs[1].set_title("Ground Truth")
        axs[2].imshow(pred_np, cmap='gray')
        axs[2].set_title(f"Predicted Mask\nDice: {dice:.4f}")
        for ax in axs:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"results/comparison_{i+1}.png", dpi=150)
        plt.close()

    print("Görseller kaydedildi: 'results/' klasörüne bak.")

if __name__ == "__main__":
    visualize_predictions()
