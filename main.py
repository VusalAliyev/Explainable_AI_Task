from src.dataset_prep import rename_images_and_update_json
from src.resolution_histogram import collect_image_resolutions_per_subset, plot_individual_histograms

if __name__ == "__main__":
    base_dir = "data"

    # 1. Dataset preparation
    for subset in ["Subset_I", "Subset_II"]:
        rename_images_and_update_json(base_dir, subset)

    # 2. Resolution histograms
    subset_res = collect_image_resolutions_per_subset(base_dir)
    plot_individual_histograms(subset_res, save_dir=".")
