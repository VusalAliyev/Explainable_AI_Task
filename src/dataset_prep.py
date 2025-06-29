# src/dataset_prep.py
import os
import json
from tqdm import tqdm

def rename_images_and_update_json(base_dir, subset):
    subset_path = os.path.join(base_dir, subset)

    for patient_id in tqdm(os.listdir(subset_path)):
        patient_path = os.path.join(subset_path, patient_id)
        if not os.path.isdir(patient_path):
            continue

        for vessel in ["LAD", "LCX", "RCA"]:
            vessel_path = os.path.join(patient_path, vessel)
            if not os.path.isdir(vessel_path):
                continue

            # JSON bul
            json_file = None
            for f in os.listdir(vessel_path):
                if f.endswith(".json"):
                    json_file = f
                    break
            if json_file is None:
                continue

            json_path = os.path.join(vessel_path, json_file)
            with open(json_path, "r") as f:
                coco = json.load(f)

            # Resimleri yeniden adlandır
            for img in coco["images"]:
                old_name = img["file_name"]
                new_name = f"{subset}_{patient_id}_{vessel}_{old_name}"
                img["file_name"] = new_name

                old_path = os.path.join(vessel_path, old_name)
                new_path = os.path.join(vessel_path, new_name)

                if os.path.exists(old_path):
                    os.rename(old_path, new_path)

            # JSON kaydet
            updated_json_path = os.path.join(vessel_path, f"{subset}_{patient_id}_{vessel}_updated.json")
            with open(updated_json_path, "w") as f:
                json.dump(coco, f)

            print(f"[✔] Updated {json_file} in {vessel_path}")


if __name__ == "__main__":
    base_dir = "data"
    for subset in ["Subset_I", "Subset_II"]:
        rename_images_and_update_json(base_dir, subset)