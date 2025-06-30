import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as T
from PIL import Image
import json
import os
from pycocotools import mask as maskUtils

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
            if ann['category_id'] in [2, 3, 4]:
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
        image_np = np.array(image).astype(np.float32) / 255.0

        mask = np.zeros((original_size[1], original_size[0]), dtype=np.uint8)
        anns = self.img_id_to_annotations.get(img_id, [])
        for ann in anns:
            rle = maskUtils.frPyObjects(ann['segmentation'], original_size[1], original_size[0])
            m = maskUtils.decode(rle)
            if len(m.shape) == 3:
                m = np.any(m, axis=2)
            mask = np.maximum(mask, m)

        mask = Image.fromarray(mask).resize((512, 512))
        mask_np = np.array(mask).astype(np.float32)

        if self.transform:
            image_tensor = self.transform(image_np)
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
            return image_tensor, mask_tensor
        return image_np, mask_np

# === Grad-CAM++ ===
class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
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
        return cam.cpu().numpy()

# === MAIN ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MiniUNet().to(device)
    model.load_state_dict(torch.load("checkpoints/trained_cotr_subsetI.pth", map_location=device))
    model.eval()

    dataset = PlaqueDataset(
        image_dir="data/Subset_I_flattened",
        annotation_file="annotations/Subset_I.json",
        transform=T.ToTensor()
    )

    idx = 0  # Görselleştirilecek örnek
    image_tensor, gt_mask = dataset[idx]
    input_tensor = image_tensor.unsqueeze(0).to(device)

    gradcam = GradCAMPlusPlus(model, target_layer=model.encoder[-1])
    cam = gradcam.generate_cam(input_tensor)

    # --- Görselleştirme ---
    image_np = image_tensor.permute(1, 2, 0).numpy()
    gt_mask_np = gt_mask.squeeze().numpy()
    cam_resized = cv2.resize(cam, (512, 512))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    overlay = heatmap + image_np
    overlay = overlay / overlay.max()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[1].imshow(gt_mask_np, cmap='gray')
    axes[1].set_title("Ground Truth")
    axes[2].imshow(overlay)
    axes[2].set_title("Grad-CAM++ Overlay")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("gradcampp_subsetI.png")
    plt.show()
