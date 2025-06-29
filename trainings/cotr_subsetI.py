import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pycocotools import mask as maskUtils

# Simple U-Net inspired architecture
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

# Dice Loss function
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice, dice.item()

# Dataset class
class PlaqueDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        with open(annotation_file) as f:
            coco = json.load(f)

        self.image_id_to_filename = {img['id']: img['file_name'] for img in coco['images']}
        self.annotations = coco['annotations']
        self.img_ids = list(self.image_id_to_filename.keys())

        self.img_id_to_annotations = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_id_to_annotations:
                self.img_id_to_annotations[img_id] = []
            self.img_id_to_annotations[img_id].append(ann)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_rel_path = self.image_id_to_filename[img_id]
        img_path = os.path.join(self.image_dir, img_rel_path)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at path: {img_path}")

        image = Image.open(img_path).convert("RGB")
        original_size = image.size  # (width, height)
        image = image.resize((512, 512))
        image = np.array(image).astype(np.float32) / 255.0

        anns = [
                    ann for ann in self.img_id_to_annotations.get(img_id, [])
                    if ann['category_id'] in [2, 3, 4]
                ]
        mask = np.zeros((original_size[1], original_size[0]), dtype=np.uint8)

        for ann in anns:
            if ann['category_id'] in [2, 3, 4]:  # Sadece plaque sınıfları
                rle = maskUtils.frPyObjects(ann['segmentation'], original_size[1], original_size[0])
                m = maskUtils.decode(rle)
                if len(m.shape) == 3:
                    m = np.any(m, axis=2)
                mask = np.maximum(mask, m)

        mask = Image.fromarray(mask.astype(np.uint8)).resize((512, 512))
        mask = np.array(mask).astype(np.float32)

        if self.transform:
            image = self.transform(image)
            mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask

# Hyperparameters and setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = T.Compose([
    T.ToTensor(),
])

dataset = PlaqueDataset(
    image_dir="data/Subset_I",
    annotation_file="annotations/Subset_I.json",
    transform=transform
)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

model = MiniUNet().to(device)
bce_loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
loss_values = []
dice_scores = []
dice_losses = []

for epoch in range(10):
    model.train()
    total_loss = 0
    total_dice_score = 0
    total_dice_loss = 0
    for images, masks in tqdm(dataloader):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)

        bce = bce_loss_fn(outputs, masks)
        dsc_loss, dsc_score = dice_loss(outputs, masks)
        loss = bce + dsc_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice_score += dsc_score
        total_dice_loss += dsc_loss.item()

    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice_score / len(dataloader)
    avg_dice_loss = total_dice_loss / len(dataloader)
    loss_values.append(avg_loss)
    dice_scores.append(avg_dice)
    dice_losses.append(avg_dice_loss)

    print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f} - Dice Score: {avg_dice:.4f} - Dice Loss: {avg_dice_loss:.4f}")

# Save model
torch.save(model.state_dict(), "trained_cotr_subsetI.pth")

# Plot Total Loss and Dice Loss
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(range(1, 11), loss_values, label='Total Loss')
ax1.set_title("Total Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True)
ax1.legend()

ax2.plot(range(1, 11), dice_losses, label='Dice Loss', color='green')
ax2.set_title("Dice Loss")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.savefig("cotr_subsetI_loss_curves.png")
plt.show()

# Plot Dice Score separately
plt.figure(figsize=(6, 4))
plt.plot(range(1, 11), dice_scores, marker='s', color='blue', label='Dice Score')
plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.title("Dice Score (CoTr - Subset I)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("cotr_subsetI_dice_score.png")
plt.show()
