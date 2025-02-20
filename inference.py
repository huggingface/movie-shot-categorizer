from transformers import AutoModelForCausalLM, AutoProcessor
import torch
from PIL import Image 
import requests 


folder_path = "/fsx/sayak/movie-shot-categorizer/model_checkpoints/epoch_9"
model = AutoModelForCausalLM.from_pretrained(
    folder_path, torch_dtype=torch.float16, trust_remote_code=True
).to("cuda").eval()
processor = AutoProcessor.from_pretrained(folder_path, trust_remote_code=True)

prompts = [
    "<COLOR>",
    "<LIGHTING>",
    "<LIGHTING_TYPE>",
    "<COMPOSITION>"
]
url = "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/shot_still.jpg"
image = Image.open(requests.get(url, stream=True).raw)

with torch.no_grad() and torch.inference_mode():
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
            generated_text,
            task=prompt,
            image_size=(image.width, image.height)
        )
        print(parsed_answer)