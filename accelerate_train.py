# Adapted from https://github.com/andimarafioti/florence2-finetuning/blob/main/distributed_train.py

import argparse
import os
from functools import partial
import logging
import torch
import math
import transformers
import diffusers
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, get_scheduler
import yaml
from types import SimpleNamespace
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, DistributedType
from data import collate_fn, get_dataset

logger = get_logger(__name__)


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    config = convert_config_types(config)
    return SimpleNamespace(**config)


def convert_config_types(obj):
    if isinstance(obj, dict):
        return {k: convert_config_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_config_types(item) for item in obj]
    elif isinstance(obj, str):
        # Try to convert to int, then float; if both fail, keep as string.
        try:
            return int(obj)
        except ValueError:
            try:
                return float(obj)
            except ValueError:
                return obj
    else:
        return obj


def create_data_loaders(
    train_dataset,
    val_dataset,
    batch_size,
    num_workers,
    processor,
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        collate_fn=partial(collate_fn, processor=processor),
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size // 2,
        pin_memory=True,
        collate_fn=partial(collate_fn, processor=processor),
        num_workers=num_workers,
    )

    return train_loader, val_loader


def forward_with_model(model, inputs, labels, weight_dtype=torch.float16):
    input_ids = inputs.input_ids
    pixel_values = inputs.pixel_values.to(weight_dtype)
    labels = labels
    return model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)


def evaluate_model(model, val_loader, device, global_step, max_val_item_count, weight_dtype, disable_pbar):
    # Evaluation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        val_item_count = 0
        for batch in tqdm(val_loader, desc=f"Evaluation at step {global_step}", disable=disable_pbar):
            val_item_count += len(batch)

            # Prepare the input and target tensors
            color_inputs, colors = batch["color_inputs"], batch["colors"]
            lighting_inputs, lightings = batch["lighting_inputs"], batch["lightings"]
            lighting_type_inputs, lighting_types = batch["lighting_type_inputs"], batch["lighting_types"]
            composition_inputs, compositions = batch["composition_inputs"], batch["compositions"]

            losses = []
            for inputs, labels in [
                (color_inputs, colors),
                (lighting_inputs, lightings),
                (lighting_type_inputs, lighting_types),
                (composition_inputs, compositions),
            ]:
                losses.append(forward_with_model(model, inputs, labels, weight_dtype=weight_dtype).loss)

            loss = torch.stack(losses).mean()

            val_loss += loss.item()
            if val_item_count > max_val_item_count:
                break

        avg_val_loss = val_loss / val_item_count

    model.train()
    return avg_val_loss


def train_model(accelerator, args):
    # Load the dataset based on the dataset_name argument
    dataset = get_dataset(
        accelerator=accelerator, dataset_id=args.dataset_id, num_proc=args.num_proc, cache_dir=args.cache_dir
    )
    with accelerator.main_process_first():
        splits = dataset.train_test_split(0.1, seed=2025)
    train_dataset, val_dataset = splits["train"], splits["test"]
    with accelerator.main_process_first():
        further_splits = val_dataset.train_test_split(0.1, seed=2025)
    val_dataset = further_splits["train"]

    # Load the model and processor
    ft_model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    # Freeze the vision tower if needed
    if args.freeze_vision_tower:
        for param in ft_model.vision_tower.parameters():
            param.requires_grad = False

    # LoRA config.
    if args.use_lora:
        TARGET_MODULES = ["q_proj", "o_proj", "k_proj", "v_proj", "linear", "Conv2d", "lm_head", "fc2"]

        config = LoraConfig(
            r=8,
            lora_alpha=8,
            target_modules=TARGET_MODULES,
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
            bias="none",
            inference_mode=False,
            use_rslora=True,
            init_lora_weights="gaussian",
        )
        ft_model = get_peft_model(ft_model, config)

    # Saving and loading hooks.
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                if isinstance(accelerator.unwrap_model(model), type(accelerator.unwrap_model(ft_model))):
                    model = accelerator.unwrap_model(model)
                    model.save_pretrained(output_dir)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

    def load_model_hook(models, input_dir):
        transformer_ = None
        if not accelerator.distributed_type == DistributedType.DEEPSPEED:
            while len(models) > 0:
                model = models.pop()

                if isinstance(accelerator.unwrap_model(model), type(accelerator.unwrap_model(ft_model))):
                    transformer_ = model  # noqa: F841
                else:
                    raise ValueError(f"unexpected save model: {accelerator.unwrap_model(model).__class__}")

        else:
            transformer_ = AutoModelForCausalLM.from_pretrained(input_dir)  # noqa: F841

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        ft_model.gradient_checkpointing_enable()

    # Create DataLoaders
    train_loader, val_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        args.batch_size,
        args.num_proc,
        processor,
    )

    # Optimizer and scheduler
    optimizer_cls = torch.optim.AdamW
    if args.use_8bit_adam:
        import bitsandbytes as bnb

        optimizer_cls = bnb.optim.AdamW8bit

    trainable_params = list(filter(lambda p: p.requires_grad, ft_model.parameters()))
    optimizer = optimizer_cls(trainable_params, lr=args.lr)

    # Math around scheduler steps and training steps.
    len_train_dataloader_after_sharding = math.ceil(len(train_loader) / accelerator.num_processes)
    num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=math.ceil(num_update_steps_per_epoch * args.epochs) * accelerator.num_processes,
    )
    ft_model, train_loader, val_loader, optimizer, lr_scheduler = accelerator.prepare(
        ft_model, train_loader, val_loader, optimizer, lr_scheduler
    )

    # Update again if needed.
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    max_train_steps = args.epochs * num_update_steps_per_epoch
    args.epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            logger.info(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            logger.info(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    # Start training!
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
    global_step = 0
    first_epoch = 0

    for epoch in range(first_epoch, args.epochs):
        # Training phase
        ft_model.train()
        for batch in train_loader:
            with accelerator.accumulate(ft_model):
                # Prepare the input and target tensors
                color_inputs, colors = batch["color_inputs"], batch["colors"]
                lighting_inputs, lightings = batch["lighting_inputs"], batch["lightings"]
                lighting_type_inputs, lighting_types = batch["lighting_type_inputs"], batch["lighting_types"]
                composition_inputs, compositions = batch["composition_inputs"], batch["compositions"]

                losses = []
                for inputs, labels in [
                    (color_inputs, colors),
                    (lighting_inputs, lightings),
                    (lighting_type_inputs, lighting_types),
                    (composition_inputs, compositions),
                ]:
                    losses.append(forward_with_model(ft_model, inputs, labels, weight_dtype=weight_dtype).loss)

                loss = torch.stack(losses).mean()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = ft_model.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
                    if global_step % args.save_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            val_loss = None
            if global_step % args.eval_steps == 0:
                val_loss = evaluate_model(
                    ft_model,
                    val_loader,
                    accelerator.device,
                    global_step,
                    args.max_val_item_count,
                    weight_dtype=weight_dtype,
                    disable_pbar=not accelerator.is_local_main_process,
                )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if val_loss:
                logs.update({"val_loss": val_loss})
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

        if global_step >= max_train_steps:
            break

        evaluate_model(
            ft_model,
            val_loader,
            accelerator.device,
            global_step,
            args.max_val_item_count,
            weight_dtype=weight_dtype,
            disable_pbar=not accelerator.is_local_main_process,
        )

    # Finish run.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(ft_model).save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)
    accelerator.end_training()


def main(args):
    model_slug = args.model_id.split("/")[-1]
    ds_slug = args.dataset_id.split("/")[-1]
    run_name = f"model@{model_slug}-ds@{ds_slug}-bs@{args.batch_size}-8bit@{args.use_8bit_adam}-lora@{args.use_lora}-lr@{args.lr}-mp@{args.mixed_precision}-fve@{args.freeze_vision_tower}"
    output_dir = Path("./model_checkpoints_accelerate") / run_name
    args.output_dir = output_dir

    accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=output_dir / "logs")
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        tracker_name = "shot-categorizer"
        accelerator.init_trackers(tracker_name, config=vars(args), init_kwargs={"wandb": {"name": run_name}})

    train_model(accelerator, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Florence-2 model on specified dataset using YAML configuration"
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the YAML configuration file.")
    args = parser.parse_args()

    config = load_config(args.config)

    main(config)
