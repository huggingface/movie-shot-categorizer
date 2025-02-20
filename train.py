# Adapted from https://github.com/andimarafioti/florence2-finetuning/blob/main/distributed_train.py

import argparse
import os
from functools import partial

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, get_scheduler

import wandb
from data import get_dataset
from peft import LoraConfig, get_peft_model


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


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


def create_data_loaders(
    train_dataset,
    val_dataset,
    batch_size,
    num_workers,
    rank,
    world_size,
    processor,
):
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=True,
        collate_fn=partial(collate_fn, processor=processor),
        num_workers=num_workers,
        sampler=train_sampler,
    )

    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size // 2,
        pin_memory=True,
        collate_fn=partial(collate_fn, processor=processor),
        num_workers=num_workers,
        sampler=val_sampler,
    )

    return train_loader, val_loader


def forward_with_model(model, inputs, labels, device):
    input_ids = inputs.input_ids.to(device, non_blocking=True)
    pixel_values = inputs.pixel_values.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    return model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)


def evaluate_model(
    rank, world_size, model, val_loader, device, train_loss, global_step, batch_size, max_val_item_count
):
    if rank == 0:
        avg_train_loss = train_loss / (global_step * batch_size * world_size)
        wandb.log({"step": global_step, "train_loss": avg_train_loss})
        print(f"Rank {rank} - Average Training Loss: {avg_train_loss}")

    # Evaluation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        val_item_count = 0
        for batch in tqdm(val_loader, desc=f"Evaluation at step {global_step}", disable=rank != 0):
            val_item_count += len(batch)

            # Prepare the input and target tensors
            color_inputs, colors = batch["color_inputs"], batch["colors"]
            lighting_inputs, lightings = batch["lighting_inputs"], batch["lightings"]
            lighting_type_inputs, lighting_types = batch["lighting_type_inputs"], batch["lighting_types"]
            composition_inputs, compositions = batch["composition_inputs"], batch["compositions"]

            losses = []
            with torch.no_grad():
                for inputs, labels in [
                    (color_inputs, colors),
                    (lighting_inputs, lightings),
                    (lighting_type_inputs, lighting_types),
                    (composition_inputs, compositions),
                ]:
                    losses.append(forward_with_model(model, inputs, labels, device=device).loss)

            loss = torch.stack(losses).mean()

            val_loss += loss.item()
            if val_item_count > max_val_item_count:
                break

        avg_val_loss = val_loss / val_item_count

        # Log metrics to wandb
        if rank == 0:
            print(f"Rank {rank} - Step {global_step} - Average Validation Loss: {avg_val_loss}")
            wandb.log({f"val_loss": avg_val_loss, "step": global_step})

    model.train()


def train_model(
    rank,
    world_size,
    model_id,
    dataset_name,
    cache_dir=None,
    num_proc=4,
    batch_size=6,
    use_lora=False,
    use_8bit_adam=False,
    epochs=10,
    lr=1e-6,
    eval_steps=10,
    max_val_item_count=1000,
    save_epochs=5,
):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # Initialize wandb
    if rank == 0:  # Only initialize wandb in the main process
        wandb.init(project="shot-categorizer")
        wandb.config.update(
            {
                "dataset": dataset_name,
                "batch_size": batch_size,
                "use_lora": use_lora,
                "epochs": epochs,
                "learning_rate": lr,
                "eval_steps": eval_steps,
                "world_size": world_size,
            }
        )

    # Load the dataset based on the dataset_name argument
    dataset = get_dataset(dataset_id=dataset_name, num_proc=num_proc, cache_dir=cache_dir)
    splits = dataset.train_test_split(0.1)
    train_dataset, val_dataset = splits["train"], splits["test"]

    # Load the model and processor
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    for param in model.vision_tower.parameters():
        param.requires_grad = False

    if use_lora:
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
        model = get_peft_model(model, config)

    model = DDP(model, device_ids=[rank])

    # Create DataLoaders
    num_workers = 0
    train_loader, val_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        batch_size,
        num_workers,
        rank,
        world_size,
        processor,
    )

    optimizer_cls = torch.optim.AdamW
    if use_8bit_adam:
        import bitsandbytes as bnb

        optimizer_cls = bnb.optim.AdamW8bit

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optimizer_cls(trainable_params, lr=lr)
    scaler = torch.amp.GradScaler(device=torch.device(device).type)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    global_step = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16):
            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}", disable=rank != 0):
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
                    losses.append(forward_with_model(model, inputs, labels, device=device).loss)

                loss = torch.stack(losses).mean()

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()

                train_loss += loss.item()
                global_step += 1

                if global_step % eval_steps == 0:
                    evaluate_model(
                        rank,
                        world_size,
                        model,
                        val_loader,
                        device,
                        train_loss,
                        global_step,
                        batch_size,
                        max_val_item_count,
                    )

        evaluate_model(
            rank, world_size, model, val_loader, device, train_loss, global_step, batch_size, max_val_item_count
        )

        # Log training loss to wandb
        avg_train_loss = train_loss / len(train_loader)
        if rank == 0:
            wandb.log({"epoch": epoch + 1, "epoch_train_loss": avg_train_loss})

        # Save model checkpoint
        if rank == 0 and global_step % save_epochs:  # Only the main process saves the checkpoint
            output_dir = f"./model_checkpoints/epoch_{epoch + 1}"
            os.makedirs(output_dir, exist_ok=True)
            model.module.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)

    # Finish the wandb run
    if rank == 0:
        model.module.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        wandb.finish()

    cleanup()


def main():
    parser = argparse.ArgumentParser(description="Train Florence-2 model on specified dataset")
    parser.add_argument("--model_id", type=str, default="microsoft/Florence-2-large")
    parser.add_argument(
        "--dataset_id", type=str, default="diffusers-internal-dev/ShotDEAD-5000", help="Dataset to train on"
    )
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--num_proc", type=int, default=4, help="Number of workers to use to process the dataset.")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size for training")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA if this flag is passed")
    parser.add_argument("--use_8bit_adam", action="store_true", help="Use 8bit Adam if this is provided.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--eval_steps", type=int, default=1000, help="Number of steps between evaluations")
    parser.add_argument(
        "--max_val_item_count", type=int, default=1000, help="Maximum number of items to evaluate on during validation"
    )
    parser.add_argument("--save_epochs", type=int, default=5, help="Checkpoint saving interval.")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(
        train_model,
        args=(
            world_size,
            args.model_id,
            args.dataset_id,
            args.cache_dir,
            args.num_proc,
            args.batch_size,
            args.use_lora,
            args.use_8bit_adam,
            args.epochs,
            args.lr,
            args.eval_steps,
            args.max_val_item_count,
            args.save_epochs,
        ),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
