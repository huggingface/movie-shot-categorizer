from transformers import AutoModelForCausalLM, AutoProcessor
import torch
from PIL import Image
import argparse


def load(repo_id):
    model = (
        AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=torch.float16, trust_remote_code=True)
        .to("cuda")
        .eval()
    )
    processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)
    return model, processor


@torch.no_grad()
@torch.inference_mode()
def infer(image_path, model, processor):
    prompts = ["<COLOR>", "<LIGHTING>", "<LIGHTING_TYPE>", "<COMPOSITION>"]
    image = Image.open(image_path)

    for prompt in prompts:
        inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda", torch.float16)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text, task=prompt, image_size=(image.width, image.height)
        )
        print(parsed_answer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, help="Path to the image.")
    parser.add_argument(
        "--repo_id", type=str, default="diffusers/shot-categorizer-v0", help="Path to the image."
    )
    args = parser.parse_args()

    model, processor = load(repo_id=args.repo_id)
    infer(image_path=args.image_path, model=model, processor=processor)
