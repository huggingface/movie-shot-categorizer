# Shot Categorizer 🎬

<div align="center">
  <img src="https://huggingface.co/diffusers/shot-categorizer-v0/resolve/main/assets/header.jpg"/>
</div>

Contains fine-tuning and inference code for a shot categorizer model. We fine-tuned the model on
the [microsoft/Florence-2-large](https://huggingface.co/microsoft/Florence-2-large) and the
[diffusers/ShotDEAD-v0](https://huggingface.co/datasets/diffusers/ShotDEAD-v0/) dataset.

* Fine-tuned model checkpoint: [diffusers/shot-categorizer-v0](https://huggingface.co/diffusers/shot-categorizer-v0)
* Dataset: [diffusers/ShotDEAD-v0](https://huggingface.co/datasets/diffusers/ShotDEAD-v0/)

The model can be [used to curate video datasets](TODO) amongst other use cases. 

## Getting started

Install PyTorch and make sure you're on a CUDA-enabled GPU. The scripts were tested on 8xH100s. 

Then install the dependencies `pip install requirements.txt`.

You can then launch training:

```bash
accelerate launch \
    --config_file=configs/accelerate-configs/ds.yaml train.py \
    --config=configs/training-configs/config.yaml
```

## Inference

```bash
python inference.py --image_path=<PATH_TO_AN_IMAGE>
```

## Acknowledgement

Thanks to [this blog post](https://huggingface.co/blog/finetune-florence2) on fine-tuning Florence-2.