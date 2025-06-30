import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch import nn

# MiniUNet tanımı (CoTr yerine kullanılıyor)
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

# Rastgele maske üretici
def generate_masks(N=1000, s=8, p1=0.5, input_size=512):
    cell_size = np.ceil(input_size / s)
    up_size = (s + 1) * int(cell_size)

    masks = []
    for _ in range(N):
        grid = np.random.choice([0, 1], size=(s, s), p=[1 - p1, p1])
        grid = cv2.resize(grid.astype(np.float32), (up_size, up_size), interpolation=cv2.INTER_NEAREST)
        x = np.random.randint(0, up_size - input_size)
        y = np.random.randint(0, up_size - input_size)
        mask = grid[y:y + input_size, x:x + input_size]
        masks.append(mask)

    return np.array(masks)

# RISE algoritması
def run_rise(model, image_path, checkpoint_path, save_path="rise_output.jpg"):
    # 🔧 Cihaz seçimi
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Görseli yükle
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Görüntü bulunamadı: {image_path}")
    img = cv2.resize(img, (512, 512))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img_tensor = torch.tensor(img_rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    # Modeli yükle
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval().to(device)

    # Maskeler
    print("→ Maskeler üretiliyor...")
    masks = generate_masks(N=500, s=8, p1=0.5, input_size=512)
    masks_torch = torch.tensor(masks).unsqueeze(1).float().to(device)

    saliency = torch.zeros((1, 1, 512, 512)).to(device)

    print("→ Maskeler uygulanıyor...")
    for i in tqdm(range(masks.shape[0])):
        masked_img = img_tensor * masks_torch[i]
        output = model(masked_img)
        score = torch.sigmoid(output).mean()
        saliency += score * masks_torch[i]

    saliency /= masks.shape[0]
    saliency = saliency.squeeze().cpu().numpy()

    # Görselleştir ve kaydet
    heatmap = show_cam_on_image(img_rgb, saliency, use_rgb=True)
    cv2.imwrite(save_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
    print("✅ RISE çıktısı kaydedildi:", save_path)

# Örnek kullanım
if __name__ == "__main__":
    image_path = "data/Subset_I_flattened/Subset_I_00004_LAD_image-00008.jpg"
    checkpoint_path = "checkpoints/trained_cotr_subsetI.pth"
    model = MiniUNet()
    run_rise(model, image_path, checkpoint_path, save_path="rise_cotr_output.jpg")
