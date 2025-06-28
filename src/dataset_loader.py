import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as T
import cv2  # segmentation mask çizimi için gerekli

class PlaqueSegmentationDataset(Dataset):
    def __init__(self, image_root, annotation_path, transform=None):
        self.image_root = image_root
        self.annotation_path = annotation_path
        self.transform = transform

        # COCO JSON dosyasını yükle
        with open(annotation_path, 'r') as f:
            self.coco = json.load(f)

        # image id -> file_name eşlemesi
        self.image_id_to_filename = {
            img['id']: img['file_name'] for img in self.coco['images']
        }

        # annotationları image_id'ye göre grupla
        self.annotations = self._group_annotations_by_image()
        self.image_ids = list(self.image_id_to_filename.keys())

    def _group_annotations_by_image(self):
        ann_dict = {}
        for ann in self.coco['annotations']:
            image_id = ann['image_id']
            if image_id not in ann_dict:
                ann_dict[image_id] = []
            ann_dict[image_id].append(ann)
        return ann_dict

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        file_name = self.image_id_to_filename[image_id]

        # Dosya yolunu normalize et (örneğin: 00004/LAD/filename.jpg)
        image_path = os.path.join(self.image_root, *file_name.split("/"))

        # Görüntüyü yükle ve maske boyutunu öğren
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        # Maske başlat
        mask = np.zeros((height, width), dtype=np.uint8)

        # Annotationları işle
        anns = self.annotations.get(image_id, [])
        for ann in anns:
            if ann["category_id"] in [2, 3, 4]:  # sadece plak kategorileri
                if 'segmentation' in ann and isinstance(ann['segmentation'], list):
                    for seg in ann['segmentation']:
                        poly = np.array(seg).reshape((-1, 2))
                        poly_int = np.round(poly).astype(np.int32)
                        cv2.fillPoly(mask, [poly_int], color=1)

        # Dönüşümler
        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)

        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask, file_name
