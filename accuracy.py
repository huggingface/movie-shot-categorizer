# git clone https://github.com/bigdata-pw/florence-tool.git; cd florence-tool; pip install -r requirements.txt; pip install -e .

from florence_tool import FlorenceTool
from datasets import load_dataset
import tqdm
import json

checkpoint = "diffusers-internal-dev/8bit_False-lora_False-lr_1e_06-mp_fp16-fve_True"
name = "8bit_False-lora_False-lr_1e_06-mp_fp16-fve_True"

florence = FlorenceTool(
    checkpoint,
    device="cuda",
    dtype="float16",
    check_task_types=False,
)
florence.load_model()

dataset = load_dataset("diffusers-internal-dev/ShotDEAD-LLTCC", streaming=True)["train"]

color = {"match": {}, "extra": {}, "missing": {}}
lighting = {"match": {}, "extra": {}, "missing": {}}
lighting_type = {"match": {}, "extra": {}, "missing": {}}
composition = {"match": {}, "extra": {}, "missing": {}}

count = 0

for still in tqdm.tqdm(dataset):
    image = still["image"].convert("RGB")
    original = {
        "<COLOR>": still["Color"],
        "<LIGHTING>": still["Lighting"],
        "<LIGHTING_TYPE>": still["Lighting Type"],
        "<COMPOSITION>": still["Composition"],
    }
    output = florence.run(
        image=image,
        task_prompt=["<COLOR>", "<LIGHTING>", "<LIGHTING_TYPE>", "<COMPOSITION>"],
    )
    output["<COLOR>"] = output["<COLOR>"].split(", ")
    output["<LIGHTING>"] = output["<LIGHTING>"].split(", ")
    output["<LIGHTING_TYPE>"] = output["<LIGHTING_TYPE>"].split(", ")
    output["<COMPOSITION>"] = output["<COMPOSITION>"].split(", ")
    matching_color = set(original["<COLOR>"]).intersection(set(output["<COLOR>"]))
    extra_color = set(output["<COLOR>"]) - set(original["<COLOR>"])
    missing_color = set(original["<COLOR>"]) - set(output["<COLOR>"])

    matching_lighting = set(original["<LIGHTING>"]).intersection(
        set(output["<LIGHTING>"])
    )
    extra_lighting = set(output["<LIGHTING>"]) - set(original["<LIGHTING>"])
    missing_lighting = set(original["<LIGHTING>"]) - set(output["<LIGHTING>"])

    matching_lighting_type = set(original["<LIGHTING_TYPE>"]).intersection(
        set(output["<LIGHTING_TYPE>"])
    )
    extra_lighting_type = set(output["<LIGHTING_TYPE>"]) - set(
        original["<LIGHTING_TYPE>"]
    )
    missing_lighting_type = set(original["<LIGHTING_TYPE>"]) - set(
        output["<LIGHTING_TYPE>"]
    )

    matching_composition = set(original["<COMPOSITION>"]).intersection(
        set(output["<COMPOSITION>"])
    )
    extra_composition = set(output["<COMPOSITION>"]) - set(original["<COMPOSITION>"])
    missing_composition = set(original["<COMPOSITION>"]) - set(output["<COMPOSITION>"])

    color["match"][still["id"]] = len(matching_color)
    color["extra"][still["id"]] = len(extra_color)
    color["missing"][still["id"]] = len(missing_color)

    lighting["match"][still["id"]] = len(matching_lighting)
    lighting["extra"][still["id"]] = len(extra_lighting)
    lighting["missing"][still["id"]] = len(missing_lighting)

    lighting_type["match"][still["id"]] = len(matching_lighting_type)
    lighting_type["extra"][still["id"]] = len(extra_lighting_type)
    lighting_type["missing"][still["id"]] = len(missing_lighting_type)

    composition["match"][still["id"]] = len(matching_composition)
    composition["extra"][still["id"]] = len(extra_composition)
    composition["missing"][still["id"]] = len(missing_composition)

    count += 1
    if count > 1000:
        break

def compute_metrics(category):
    total_match = sum(category["match"].values())
    total_missing = sum(category["missing"].values())
    total_extra = sum(category["extra"].values())

    total = total_match + total_missing + total_extra

    precision = total_match / (total_match + total_extra) if (total_match + total_extra) > 0 else 0
    recall = total_match / (total_match + total_missing) if (total_match + total_missing) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (total_match / total) if total > 0 else 0

    return {
        'precision': precision * 100,
        "recall": recall * 100,
        "f1_score": f1_score * 100,
        "accuracy": accuracy * 100
    }


color_metrics = compute_metrics(color)
lighting_metrics = compute_metrics(lighting)
lighting_type_metrics = compute_metrics(lighting_type)
composition_metrics = compute_metrics(composition)

overall_precision = (color_metrics['precision'] + lighting_metrics['precision'] + lighting_type_metrics['precision'] + composition_metrics['precision']) / 4
overall_recall = (color_metrics['recall'] + lighting_metrics['recall'] + lighting_type_metrics['recall'] + composition_metrics['recall']) / 4
overall_f1 = (color_metrics['f1_score'] + lighting_metrics['f1_score'] + lighting_type_metrics['f1_score'] + composition_metrics['f1_score']) / 4
overall_accuracy = (color_metrics['accuracy'] + lighting_metrics['accuracy'] + lighting_type_metrics['accuracy'] + composition_metrics['accuracy']) / 4

print(f"Color - Precision: {color_metrics['precision']:.2f}%, Recall: {color_metrics['recall']:.2f}%, F1-score: {color_metrics['f1_score']:.2f}%, Accuracy-score: {color_metrics['accuracy']:.2f}%")
print(f"Lighting - Precision: {lighting_metrics['precision']:.2f}%, Recall: {lighting_metrics['recall']:.2f}%, F1-score: {lighting_metrics['f1_score']:.2f}%, Accuracy-score: {lighting_metrics['accuracy']:.2f}%")
print(f"Lighting Type - Precision: {lighting_type_metrics['precision']:.2f}%, Recall: {lighting_type_metrics['recall']:.2f}%, F1-score: {lighting_type_metrics['f1_score']:.2f}%, Accuracy-score: {lighting_type_metrics['accuracy']:.2f}%")
print(f"Composition - Precision: {composition_metrics['precision']:.2f}%, Recall: {composition_metrics['recall']:.2f}%, F1-score: {composition_metrics['f1_score']:.2f}%, Accuracy-score: {composition_metrics['accuracy']:.2f}%")
print(f"Overall - Precision: {overall_precision:.2f}%, Recall: {overall_recall:.2f}%, F1-score: {overall_f1:.2f}%, Accuracy-score: {overall_accuracy:.2f}%")

with open(f"{name}.json", "w") as f:
    json.dump(
        {
            "color": color,
            "lighting": lighting,
            "lighting_type": lighting_type,
            "composition": composition,
            "color_metrics": color_metrics,
            "lighting_metrics": lighting_metrics,
            "lighting_type_metrics": lighting_type_metrics,
            "composition_metrics": composition_metrics,
            "overall_precision": overall_precision,
            "overall_recall": overall_recall,
            "overall_f1": overall_f1,
            "overall_accuracy": overall_accuracy,
        }
    )
