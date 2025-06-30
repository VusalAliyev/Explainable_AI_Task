import os
import shutil
from tqdm import tqdm

def flatten_images(base_dir, subset):
    source_dir = os.path.join(base_dir, subset)
    dest_dir = os.path.join(base_dir, subset + "_flattened")

    os.makedirs(dest_dir, exist_ok=True)

    for patient_id in tqdm(os.listdir(source_dir), desc=f"Processing {subset}"):
        patient_path = os.path.join(source_dir, patient_id)
        if not os.path.isdir(patient_path):
            continue

        for vessel in ["LAD", "LCX", "RCA"]:
            vessel_path = os.path.join(patient_path, vessel)
            if not os.path.isdir(vessel_path):
                continue

            for fname in os.listdir(vessel_path):
                if fname.endswith(".jpg") and fname.startswith(subset):
                    src = os.path.join(vessel_path, fname)
                    dst = os.path.join(dest_dir, fname)
                    shutil.copy2(src, dst)  # use move() instead if you want to *move* them
    print(f"[✔] All images copied to {dest_dir}")

if __name__ == "__main__":
    base_dir = "data"
    flatten_images(base_dir, "Subset_I")
    flatten_images(base_dir, "Subset_II")
