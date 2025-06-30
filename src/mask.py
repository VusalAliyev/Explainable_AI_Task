import os
import json
import numpy as np
import cv2
from pycocotools import mask as maskUtils
from PIL import Image
from tqdm import tqdm

def load_annotations(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def generate_masks(subset_name):
    # Path configs
    image_dir = os.path.join("data", subset_name)
    annotation_path = os.path.join("annotations", f"{subset_name.lower()}.json")
    output_mask_dir = os.path.join("masks", subset_name)
    os.makedirs(output_mask_dir, exist_ok=True)

    data = load_annotations(annotation_path)
    image_id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

    plaque_category_ids = {2, 3, 4}  # only plaque types
    masks_per_image = {}

    for ann in data["annotations"]:
        if ann["category_id"] not in plaque_category_ids:
            continue

        image_id = ann["image_id"]
        segmentation = ann["segmentation"]

        if image_id not in masks_per_image:
            img_info = next((img for img in data["images"] if img["id"] == image_id), None)
            if img_info is None:
                continue
            height = img_info["height"]
            width = img_info["width"]
            masks_per_image[image_id] = np.zeros((height, width), dtype=np.uint8)

        if isinstance(segmentation, list):
            # Polygon format
            rles = maskUtils.frPyObjects(segmentation, height, width)
            rle = maskUtils.merge(rles)
        else:
            rle = segmentation

        binary_mask = maskUtils.decode(rle)
        masks_per_image[image_id] = np.maximum(masks_per_image[image_id], binary_mask * 255)

    for image_id, mask in tqdm(masks_per_image.items(), desc=f"Saving masks for {subset_name}"):
        filename = image_id_to_filename[image_id]
        mask_path = os.path.join(output_mask_dir, filename.replace(".jpg", ".png"))
        Image.fromarray(mask).save(mask_path)

    print(f"[✓] Masks saved to {output_mask_dir}")


if __name__ == "__main__":
    generate_masks("Subset_I")