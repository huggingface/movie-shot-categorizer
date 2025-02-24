from datasets import load_dataset
import accelerate 


NONE_KEY_MAP = {
    "Color": ["no_color"],
    "Lighting": ["no_lighting"],
    "Lighting Type": ["no_ligting_type"],
    "Composition": ["no_composition"],
}


def preprocess_batch(rows):
    """
    Process a batch of examples represented as a dictionary where each key maps to a list of values.
    For each key in NONE_KEY_MAP, create a new column with processed details.
    Also, conditionally convert images to RGB if they are not already.
    """
    n = len(next(iter(rows.values())))

    # Prepare storage for new processed columns.
    # For each key in NONE_KEY_MAP, the new key is defined as: <{original_key with spaces replaced, uppercase}>
    processed_data = {}
    for k in NONE_KEY_MAP:
        new_key = f"<{k.replace(' ', '_').upper()}>"
        processed_data[new_key] = []
    
    # Process each example (by index)
    for i in range(n):
        # For each key in NONE_KEY_MAP, process the value for the i-th example.
        for k in NONE_KEY_MAP:
            # If the key is missing, we assume a list of Nones.
            value = rows.get(k, [None] * n)[i]
            if value:
                if isinstance(value, list):
                    detail = ", ".join(value)
                else:
                    detail = str(value)
            else:
                default = NONE_KEY_MAP[k][0]
                detail = default.replace("no_", "unspecified ").replace("_", " ")
            new_key = f"<{k.replace(' ', '_').upper()}>"
            processed_data[new_key].append(detail)
        
        # Process the image field if present.
        if "image" in rows:
            image = rows["image"][i]
            if image is not None and hasattr(image, "mode"):
                if image.mode != "RGB":
                    image = image.convert("RGB")
            rows["image"][i] = image

    # Merge the processed columns into the original batch dictionary.
    rows.update(processed_data)
    return rows


def get_dataset(accelerator=None, dataset_id="diffusers-internal-dev/ShotDEAD-5000", cache_dir=None, num_proc=4):
    if accelerator is None:
        accelerator = accelerate.PartialState()
    
    keep_cols = set(["Color", "image", "Lighting", "Lighting Type", "Composition"])
    dataset = load_dataset(dataset_id, split="train", cache_dir=cache_dir)
    all_cols = set(dataset.features.keys())
    dataset = dataset.remove_columns(list(all_cols - keep_cols))
    
    with accelerator.main_process_first():
        dataset = dataset.with_transform(preprocess_batch)
    return dataset


def collate_fn(batch, processor, max_length=800):
    images = [sample["image"] for sample in batch]

    # Map each field to its corresponding key.
    field_map = {
        "color": "<COLOR>",
        "lighting": "<LIGHTING>",
        "lighting_type": "<LIGHTING_TYPE>",
        "composition": "<COMPOSITION>",
    }

    collated = {}
    for name, key in field_map.items():
        # Create a list of placeholder prompts and extract the actual text from each sample.
        prompts = [key] * len(batch)
        texts = [sample[key] for sample in batch]

        # Tokenize the raw texts.
        tokenized = processor.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_token_type_ids=False,
        ).input_ids

        # Process the images along with the placeholder prompts.
        processed_inputs = processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        # Store the processed inputs and tokenized texts using consistent naming.
        collated[f"{name}_inputs"] = processed_inputs
        if name == "color":
            collated["colors"] = tokenized
        elif name == "lighting":
            collated["lightings"] = tokenized
        elif name == "lighting_type":
            collated["lighting_types"] = tokenized
        elif name == "composition":
            collated["compositions"] = tokenized

    return collated