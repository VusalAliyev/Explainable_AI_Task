import os
from PIL import Image
import matplotlib.pyplot as plt

def collect_image_sizes(root_dir):
    widths = []
    heights = []
    for subset in os.listdir(root_dir):
        subset_path = os.path.join(root_dir, subset)
        if not os.path.isdir(subset_path):
            continue
        for patient in os.listdir(subset_path):
            for vessel in ['LAD', 'LCX', 'RCA']:
                image_dir = os.path.join(subset_path, patient, vessel)
                if not os.path.exists(image_dir):
                    continue
                for fname in os.listdir(image_dir):
                    if fname.endswith('.jpg'):
                        img_path = os.path.join(image_dir, fname)
                        try:
                            with Image.open(img_path) as img:
                                width, height = img.size
                                widths.append(width)
                                heights.append(height)
                        except Exception as e:
                            print(f"Could not open {img_path}: {e}")
    return widths, heights

def plot_resolution_histogram(widths, heights, save_path='width_height_distribution.png'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(widths, bins=30)
    axes[0].set_title("Image Width Distribution")
    axes[0].set_xlabel("Width (pixels)")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(heights, bins=30)
    axes[1].set_title("Image Height Distribution")
    axes[1].set_xlabel("Height (pixels)")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[✔] Saved to {save_path}")

if __name__ == "__main__":
    root = "data"
    w, h = collect_image_sizes(root)
    plot_resolution_histogram(w, h)
