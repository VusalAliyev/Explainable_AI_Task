import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import json
import os
from pycocotools import mask as maskUtils

# === MODEL VE DATASET SINIFI ===

class MiniUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

class PlaqueDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        with open(annotation_file) as f:
            coco = json.load(f)

        self.image_id_to_filename = {img['id']: img['file_name'] for img in coco['images']}
        self.img_ids = list(self.image_id_to_filename.keys())

        self.img_id_to_annotations = {}
        for ann in coco['annotations']:
            if ann['category_id'] in [2, 3, 4]:  # SADECE PLAQUE
                img_id = ann['image_id']
                if img_id not in self.img_id_to_annotations:
                    self.img_id_to_annotations[img_id] = []
                self.img_id_to_annotations[img_id].append(ann)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.image_dir, self.image_id_to_filename[img_id])
        image = Image.open(img_path).convert("RGB")
        original_size = image.size
        image = image.resize((512, 512))
        image = np.array(image).astype(np.float32) / 255.0

        mask = np.zeros((original_size[1], original_size[0]), dtype=np.uint8)
        anns = self.img_id_to_annotations.get(img_id, [])

        for ann in anns:
            rle = maskUtils.frPyObjects(ann['segmentation'], original_size[1], original_size[0])
            m = maskUtils.decode(rle)
            if len(m.shape) == 3:
                m = np.any(m, axis=2)
            mask = np.maximum(mask, m)

        mask = Image.fromarray(mask).resize((512, 512))
        mask = np.array(mask).astype(np.float32)

        if self.transform:
            image = self.transform(image)
            mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask

# === BAŞLANGIÇ ===

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = T.Compose([T.ToTensor()])

dataset = PlaqueDataset(
    image_dir="data/Subset_II",
    annotation_file="annotations/Subset_II.json",
    transform=transform
)

model = MiniUNet().to(device)
model.load_state_dict(torch.load("trained_cotr_subsetII.pth", map_location=device))
model.eval()

# === GÖRSELLEŞTİRME ===
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

    intersection = (pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()
    dice_score = (2 * intersection + 1e-6) / (union + 1e-6)

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
    plt.savefig(f"subsetII_visualize_sample_{idx}.png")
    plt.show()
