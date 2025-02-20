from datasets import load_dataset
import torch.distributed as dist

NONE_KEY_MAP = {
    "Color": ["no_color"],
    "Lighting": ["no_lighting"],
    "Lighting Type": ["no_ligting_type"],
    "Composition": ["no_composition"],
}


def preprocess(row):
    def get_detail(key):
        value = row[key]
        if value:
            if isinstance(value, list):
                # Join multiple values into a comma-separated string.
                return ", ".join(value)
            return str(value)
        else:
            # Use a fallback default, optionally replacing placeholders to sound more natural.
            default = NONE_KEY_MAP[key][0]
            return default.replace("no_", "unspecified ").replace("_", " ")

    for k in NONE_KEY_MAP:
        processed_k = k.replace(" ", "_").upper()
        row[f"<{processed_k}>"] = get_detail(k)

    row["image"] = row["image"].convert("RGB")
    return row


def get_dataset(dataset_id="diffusers-internal-dev/ShotDEAD-5000", cache_dir=None, num_proc=4):
    keep_cols = set(["Color", "image", "Lighting", "Lighting Type", "Composition"])
    dataset = load_dataset(dataset_id, split="train", cache_dir=cache_dir)
    all_cols = set(dataset.features.keys())
    dataset = dataset.remove_columns(list(all_cols - keep_cols))
    dataset = dataset.map(preprocess, num_proc=num_proc)
    return dataset
