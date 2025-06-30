import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import segmentation_models_pytorch as smp
from tqdm import tqdm

# === Ayarlar ===
IMAGE_DIR = "data/Subset_I_flattened"
MASK_DIR = "masks/Subset_I"
MODEL_PATH = "checkpoints/pspnet_subset_i.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256
N_MASKS = 2000         
MASK_PROB = 0.3      
RESIZE_MASK = 16       

# === Dataset ===
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
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img_tensor = torch.tensor(np.transpose(img_rgb, (2, 0, 1)), dtype=torch.float32)

        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        mask = (mask > 0).astype(np.float32)
        mask_tensor = torch.tensor(mask).unsqueeze(0)

        return img_tensor, mask_tensor, img_rgb, mask

# === RISE maskeleri üret ===
def generate_masks(n_masks, mask_size, p=0.5):
    masks = np.random.binomial(1, p, size=(n_masks, mask_size, mask_size)).astype(np.float32)
    masks = np.array([cv2.resize(m, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR) for m in masks])
    return masks[:, np.newaxis, :, :]  # (N,1,H,W)

# === RISE hesapla ===
def rise_explain(model, input_tensor, masks):
    model.eval()
    input_tensor = input_tensor.to(DEVICE)

    heatmap = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    for i in tqdm(range(len(masks))):
        mask = torch.tensor(masks[i]).to(DEVICE)  # (1,1,H,W)
        masked_input = input_tensor * mask
        masked_input = masked_input.expand(-1, 3, -1, -1)

        with torch.no_grad():
            output = model(masked_input)
            score = output.sigmoid().mean().item()

        heatmap += masks[i, 0] * score

    heatmap = heatmap / np.max(heatmap)
    return heatmap

# === Görselleştir ===
def visualize_rise():
    os.makedirs("rise_results", exist_ok=True)

    model = smp.PSPNet(encoder_name="resnet34", in_channels=3, classes=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    dataset = SegDataset(IMAGE_DIR, MASK_DIR)
    masks = generate_masks(N_MASKS, RESIZE_MASK, p=MASK_PROB)

    for i in range(5):
        img_tensor, _, img_rgb, gt_mask = dataset[i]
        input_tensor = img_tensor.unsqueeze(0)

        heatmap = rise_explain(model, input_tensor, masks)

        # RISE Heatmap'i oluştur
        cam_overlay = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        cam_overlay = cv2.cvtColor(cam_overlay, cv2.COLOR_BGR2RGB)
        overlay = (img_rgb * 255).astype(np.uint8)
        combined = cv2.addWeighted(overlay, 0.5, cam_overlay, 0.5, 0)

        # Triple Görsel
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(img_rgb)
        axs[0].set_title("Original Image")
        axs[1].imshow(gt_mask, cmap='gray')
        axs[1].set_title("Ground Truth")
        axs[2].imshow(combined)
        axs[2].set_title("RISE Heatmap")
        for ax in axs:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"rise_results/rise_viz_{i+1}.png", dpi=150)
        plt.close()

    print("✅ RISE görselleri başarıyla 'rise_results/' klasörüne kaydedildi.")

# === Ana ===
if __name__ == "__main__":
    visualize_rise()
