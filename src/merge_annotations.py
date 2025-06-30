import os
import json
import glob
from tqdm import tqdm

def merge_subset_jsons(subset_path, output_path):
    image_id_counter = 1
    annotation_id_counter = 1
    merged = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    image_id_map = {}

    # _updated.json dosyalarını topla
    all_jsons = glob.glob(os.path.join(subset_path, "*", "*", "*_updated.json"))
    print(f"[🔍] Found {len(all_jsons)} updated JSON files in {subset_path}")

    for json_file in tqdm(all_jsons, desc=f"Merging {subset_path}"):
        with open(json_file, 'r') as f:
            data = json.load(f)

        if not merged["categories"]:
            merged["categories"] = data.get("categories", [])

        for img in data["images"]:
            old_id = img["id"]
            original_name = img["file_name"]

            # Artık sadece filename olacak (alt klasör yok)
            new_img = img.copy()
            new_img["file_name"] = os.path.basename(original_name)
            new_img["id"] = image_id_counter

            image_id_map[old_id] = image_id_counter
            merged["images"].append(new_img)
            image_id_counter += 1

        for ann in data["annotations"]:
            new_ann = ann.copy()
            new_ann["id"] = annotation_id_counter
            new_ann["image_id"] = image_id_map[ann["image_id"]]
            merged["annotations"].append(new_ann)
            annotation_id_counter += 1

    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2)
    print(f"[✔] Merged JSON saved to {output_path}")

def main():
    os.makedirs("annotations", exist_ok=True)

    # Merge Subset I
    merge_subset_jsons("data/Subset_I", "annotations/Subset_I.json")

    # Merge Subset II
    merge_subset_jsons("data/Subset_II", "annotations/Subset_II.json")

    # Combine both into a single Combined.json
    with open("annotations/Subset_I.json", "r") as f1, open("annotations/Subset_II.json", "r") as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    combined = {
        "images": [],
        "annotations": [],
        "categories": data1["categories"]  # assume same across subsets
    }

    image_id_counter = 1
    annotation_id_counter = 1
    image_id_map = {}

    for img in data1["images"] + data2["images"]:
        new_img = img.copy()
        old_id = new_img["id"]
        new_img["id"] = image_id_counter
        image_id_map[old_id] = image_id_counter
        combined["images"].append(new_img)
        image_id_counter += 1

    for ann in data1["annotations"] + data2["annotations"]:
        new_ann = ann.copy()
        new_ann["id"] = annotation_id_counter
        new_ann["image_id"] = image_id_map[ann["image_id"]]
        combined["annotations"].append(new_ann)
        annotation_id_counter += 1

    with open("annotations/Combined.json", "w") as f:
        json.dump(combined, f, indent=2)
    print("[✔] Combined JSON saved to annotations/Combined.json")

if __name__ == "__main__":
    main()
