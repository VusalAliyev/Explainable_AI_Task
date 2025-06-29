import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import json
import numpy as np
from PIL import Image
from pycocotools import mask as maskUtils
import torchvision.transforms as T

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

        image = Image.open(img_path).convert("RGB")
        original_size = image.size
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
