import os
import matplotlib.pyplot as plt
from src.dataset_loader import PlaqueSegmentationDataset

# Paths
data_root = "data/Subset_I"  # You can switch to Subset_II later
annotation_path = "annotations/Subset_I.json"

# Load dataset
dataset = PlaqueSegmentationDataset(image_root=data_root, annotation_path=annotation_path)

# Visualize and save 3 samples
os.makedirs("visuals", exist_ok=True)

for idx in [0, 1, 2]:
    image, mask, filename = dataset[idx]
    image_np = image.permute(1, 2, 0).numpy()
    mask_np = mask.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image_np)
    axes[0].set_title("Preprocessed Image")
    axes[0].axis("off")

    axes[1].imshow(mask_np, cmap="gray")
    axes[1].set_title("Segmentation Mask")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(f"visuals/sample_{idx+1}.png")
    plt.close()
