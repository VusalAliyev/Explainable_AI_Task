import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from pycocotools import mask as maskUtils
import torchvision.transforms as T
from torch.nn import functional as F


# === MODEL ===
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


# === DATASET ===
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


# === GradCAM++ ===
class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)
        output = torch.sigmoid(output)
        output.mean().backward()

        grads = self.gradients
        acts = self.activations

        grads_power_2 = grads ** 2
        grads_power_3 = grads ** 3

        sum_acts = torch.sum(acts, dim=(2, 3), keepdim=True)
        eps = 1e-8
        alpha_num = grads_power_2
        alpha_denom = 2 * grads_power_2 + sum_acts * grads_power_3
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.tensor(eps).to(alpha_denom.device))
        alphas = alpha_num / alpha_denom

        weights = torch.sum(alphas * torch.relu(grads), dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * acts, dim=1).squeeze()

        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-6)
        cam = cam.cpu().numpy()
        return cam


# === MAIN ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MiniUNet().to(device)
    model.load_state_dict(torch.load("trained_cotr_subsetII.pth", map_location=device))
    model.eval()

    dataset = PlaqueDataset(
        image_dir="data/Subset_II",
        annotation_file="annotations/Subset_II.json",
        transform=T.ToTensor()
    )

    idx = 0  # Dilersen değiştir
    image, mask = dataset[idx]
    input_tensor = image.unsqueeze(0).to(device)

    target_layer = model.encoder[-1]  # Son ReLU sonrası Conv layer
    campp = GradCAMPlusPlus(model, target_layer)
    cam = campp.generate_cam(input_tensor)

    input_np = image.permute(1, 2, 0).numpy()
    cam_resized = cv2.resize(cam, (512, 512))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    overlay = heatmap + input_np
    overlay = overlay / overlay.max()

    # === Görselleştirme ===
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(input_np)
    plt.title("Input Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cam_resized, cmap='jet')
    plt.title("Grad-CAM++")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("visuals/gradcampp_subsetII.png")
    plt.show()
