import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

# Ayarlar
IMAGE_DIR = "data/Subset_I_flattened"
MASK_DIR = "masks/Subset_I"
SAVE_MODEL_PATH = "pspnet_subset_i.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 20  
BATCH_SIZE = 4
LR = 1e-5    # Daha küçük öğrenme oranı
IMG_SIZE = 256

# Dataset sınıfı
class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.imgs = []
        self.masks = []

        for mask_file in sorted(os.listdir(mask_dir)):
            if not mask_file.endswith(".png"):
                continue
            img_file = mask_file.replace(".png", ".jpg")
            img_path = os.path.join(img_dir, img_file)
            mask_path = os.path.join(mask_dir, mask_file)

            if os.path.exists(img_path) and os.path.exists(mask_path):
                img = cv2.imread(img_path)
                if img is not None:
                    self.imgs.append(img_file)
                    self.masks.append(mask_file)

        print(f"Toplam {len(self.imgs)} geçerli örnek bulundu.")

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

# Boundary Loss
class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        smooth = 1.
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return 1 - dice + self.bce(pred, target)

# Dice skoru
def dice_score(pred, mask):
    pred = (pred > 0.5).float()
    smooth = 1e-6
    intersection = (pred * mask).sum()
    return (2. * intersection + smooth) / (pred.sum() + mask.sum() + smooth)

# Eğitim
def train():
    train_dataset = SegDataset(IMAGE_DIR, MASK_DIR)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = smp.PSPNet(encoder_name="resnet34", in_channels=3, classes=1).to(DEVICE)
    criterion = BoundaryLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    loss_list = []
    dice_list = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_dice = 0.0

        for imgs, masks in tqdm(train_loader):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            with torch.no_grad():
                running_dice += dice_score(torch.sigmoid(preds), masks).item()

        avg_loss = running_loss / len(train_loader)
        avg_dice = running_dice / len(train_loader)
        loss_list.append(avg_loss)
        dice_list.append(avg_dice)

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}")

    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print("Model kayıt edildi:", SAVE_MODEL_PATH)

    # Loss plot
    plt.figure()
    plt.plot(range(1, EPOCHS+1), loss_list, label='Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Boundary Loss")
    plt.legend()
    plt.savefig("loss_plot.png")

    # Dice plot
    plt.figure()
    plt.plot(range(1, EPOCHS+1), dice_list, label='Dice Score', color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Dice per Epoch")
    plt.legend()
    plt.savefig("dice_plot.png")
    plt.show()

# Tahmin ve görsel kaydetme
def visualize():
    os.makedirs("results", exist_ok=True)

    model = smp.PSPNet(encoder_name="resnet34", in_channels=3, classes=1).to(DEVICE)
    model.load_state_dict(torch.load(SAVE_MODEL_PATH))
    model.eval()

    dataset = SegDataset(IMAGE_DIR, MASK_DIR)

    for i in range(7):
        img, mask = dataset[i]
        img_tensor = img.unsqueeze(0).to(DEVICE)
        mask = mask.to(DEVICE)
        with torch.no_grad():
            pred = model(img_tensor)
        pred = torch.sigmoid(pred)
        dice = dice_score(pred, mask).item()

        img_np = img_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
        mask_np = mask.squeeze().cpu().numpy()
        pred_np = pred.squeeze().cpu().numpy() > 0.5

        fig, axs = plt.subplots(1, 3, figsize=(10, 3))
        axs[0].imshow(img_np)
        axs[0].set_title("Image")
        axs[1].imshow(mask_np, cmap='gray')
        axs[1].set_title("Ground Truth")
        axs[2].imshow(pred_np, cmap='gray')
        axs[2].set_title(f"Prediction\nDice: {dice:.4f}")
        for ax in axs:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"results/prediction_{i+1}.jpg")
        plt.close()

if __name__ == "__main__":
    train()
    visualize()