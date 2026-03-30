"""
this file has helper fcts, the main training fct, checkpointing and uploading to huggingface fct 

Sources:
- nanoGPT (Karpathy):
  training loop structure, gradient accumulation, cosine LR schedule
  https://github.com/karpathy/nanoGPT/blob/master/train.py

- PyTorch training:
  optimizer step, backward pass, gradient clipping
  https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

- PyTorch AMP (mixed precision):
  torch.autocast, GradScaler
  https://pytorch.org/docs/stable/amp.html

- Hugging Face Transformers:
  tokenizer loading (GPT-2 tokenizer)
  https://huggingface.co/docs/transformers/main_classes/tokenizer

- Hugging Face Hub:
  model upload via upload_folder
  https://huggingface.co/docs/huggingface_hub/guides/upload

- Learning rate scheduling:
  cosine decay with warmup 
"""

import os
import math
import json
import time
import argparse
from contextlib import nullcontext

import torch
from transformers import AutoTokenizer
from huggingface_hub import login, upload_folder

from model import GPT, GPTConfig
from data import build_train_dataloader, estimate_num_steps


def parse_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--target_tokens", type=int, default=5_000_000_000)
    parser.add_argument("--shuffle_buffer", type=int, default=10_000)
    parser.add_argument("--num_workers", type=int, default=0)

    # model
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--n_head", type=int, default=12)
    parser.add_argument("--n_embd", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--no_bias", dest="bias", action="store_false")
    parser.set_defaults(bias=True)

    parser.add_argument("--flash", action="store_true")
    parser.add_argument("--no_flash", dest="flash", action="store_false")
    parser.set_defaults(flash=True)

    parser.add_argument("--sparse_block_size", type=int, default=64)

    # training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    # precision
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float32", "float16", "bfloat16"])

    # logging/checkpointing
    parser.add_argument("--output_dir", type=str, default="outputs/run1")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--save_final_only", action="store_true")

    # resume
    parser.add_argument("--resume_from", type=str, default=None)

    # hub upload
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hf_repo_id", type=str, default=None)
    parser.add_argument("--hf_private", action="store_true")
    parser.add_argument("--hf_token", type=str, default=None)

    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_amp_context(device: str, dtype_str: str):
    if device != "cuda":
        return nullcontext(), None

    if dtype_str == "float16":
        return torch.autocast(device_type="cuda", dtype=torch.float16), torch.float16
    if dtype_str == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16), torch.bfloat16
    return nullcontext(), None



def cosine_lr(step, max_steps, warmup_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr

    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


def save_checkpoint(
    output_dir,
    step,
    model,
    optimizer,
    args,
    config,
    tokenizer,
    scaler=None,
    final_name=None,
):
    ckpt_dir = os.path.join(
        output_dir,
        final_name if final_name is not None else f"checkpoint_step_{step}"
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    # model weights
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "pytorch_model.bin"))

    # optimizer/scaler/training state
    train_state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }
    if scaler is not None:
        train_state["scaler"] = scaler.state_dict()

    torch.save(train_state, os.path.join(ckpt_dir, "trainer_state.pt"))

    # config/tokenizer
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(config.__dict__, f, indent=2)

    tokenizer.save_pretrained(ckpt_dir)

    # simple model card
    readme_path = os.path.join(ckpt_dir, "README.md")
    if not os.path.exists(readme_path):
        with open(readme_path, "w") as f:
            f.write(
                f"# Custom GPT model\n\n"
                f"- tokenizer: {args.tokenizer_name}\n"
                f"- block_size: {args.block_size}\n"
                f"- n_layer: {args.n_layer}\n"
                f"- n_head: {args.n_head}\n"
                f"- n_embd: {args.n_embd}\n"
                f"- target_tokens: {args.target_tokens}\n"
                f"- step: {step}\n"
            )

    return ckpt_dir


def maybe_resume(model, optimizer, scaler, resume_from, device):
    trainer_state_path = os.path.join(resume_from, "trainer_state.pt")
    model_path = os.path.join(resume_from, "pytorch_model.bin")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
    if not os.path.exists(trainer_state_path):
        raise FileNotFoundError(f"Missing trainer state: {trainer_state_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    trainer_state = torch.load(trainer_state_path, map_location=device)

    optimizer.load_state_dict(trainer_state["optimizer"])
    if scaler is not None and "scaler" in trainer_state:
        scaler.load_state_dict(trainer_state["scaler"])

    start_step = trainer_state.get("step", 0)
    return start_step


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    # match model.py vocab assumption
    vocab_size = tokenizer.vocab_size

    config = GPTConfig(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        d_ffn=4 * args.n_embd,
        n_ctx=args.block_size,
        vocab_size=vocab_size,
        dropout=args.dropout,
        bias=args.bias,
        flash=args.flash,
        sparse_block_size=args.sparse_block_size,
    )

    model = GPT(config).to(device)

    optimizer = model.configure_optimizers(
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        device_type=device,
    )

    amp_context, amp_dtype = get_amp_context(device, args.dtype)
    use_grad_scaler = device == "cuda" and args.dtype == "float16"
    scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)

    train_loader = build_train_dataloader(
        tokenizer_name=args.tokenizer_name,
        block_size=args.block_size,
        batch_size=args.batch_size,
        target_tokens=args.target_tokens,
        shuffle_buffer=args.shuffle_buffer,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    estimated_steps = estimate_num_steps(
        target_tokens=args.target_tokens,
        block_size=args.block_size,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
    )

    max_steps = args.max_steps if args.max_steps is not None else estimated_steps

    print(f"Estimated steps from token budget: {estimated_steps}")
    print(f"Training max_steps: {max_steps}")

    start_step = 0
    if args.resume_from:
        print(f"Resuming from: {args.resume_from}")
        start_step = maybe_resume(model, optimizer, scaler, args.resume_from, device)
        print(f"Resumed at step: {start_step}")

    model.train()
    running_loss = 0.0
    running_tokens = 0
    global_step = start_step
    t0 = time.time()

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        print(f"Starting epoch {epoch + 1}/{args.epochs}")

        for batch_idx, batch in enumerate(train_loader):
            if global_step >= max_steps:
                break

            idx = batch["idx"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)

            # set LR per step
            lr = cosine_lr(
                step=global_step,
                max_steps=max_steps,
                warmup_steps=args.warmup_steps,
                max_lr=args.lr,
                min_lr=args.min_lr,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            with amp_context:
                logits, loss = model(idx, targets)
                loss = loss / args.grad_accum_steps

            if use_grad_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            should_step = ((batch_idx + 1) % args.grad_accum_steps == 0)

            if should_step:
                if args.grad_clip is not None and args.grad_clip > 0:
                    if use_grad_scaler:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                if use_grad_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                global_step += 1

                loss_item = loss.item() * args.grad_accum_steps
                running_loss += loss_item
                running_tokens += idx.numel() * args.grad_accum_steps

                if global_step % args.log_interval == 0:
                    elapsed = time.time() - t0
                    avg_loss = running_loss / args.log_interval
                    toks_per_sec = running_tokens / max(elapsed, 1e-8)
                    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")

                    print(
                        f"step {global_step}/{max_steps} | "
                        f"lr {lr:.6e} | "
                        f"loss {avg_loss:.4f} | "
                        f"ppl {ppl:.2f} | "
                        f"tok/s {toks_per_sec:.0f}"
                    )

                    running_loss = 0.0
                    running_tokens = 0
                    t0 = time.time()

                if (not args.save_final_only) and (global_step % args.save_interval == 0):
                    ckpt_dir = save_checkpoint(
                        output_dir=args.output_dir,
                        step=global_step,
                        model=model,
                        optimizer=optimizer,
                        args=args,
                        config=config,
                        tokenizer=tokenizer,
                        scaler=scaler if use_grad_scaler else None,
                    )
                    print(f"Saved checkpoint to {ckpt_dir}")

            if global_step >= max_steps:
                break

        if global_step >= max_steps:
            break

    final_dir = save_checkpoint(
        output_dir=args.output_dir,
        step=global_step,
        model=model,
        optimizer=optimizer,
        args=args,
        config=config,
        tokenizer=tokenizer,
        scaler=scaler if use_grad_scaler else None,
        final_name="final_model",
    )
    print(f"Saved final model to {final_dir}")

    if args.push_to_hub: #TODO double check good security practices for this .. 
        if not args.hf_repo_id:
            raise ValueError("--hf_repo_id is required when --push_to_hub is set")

        if args.hf_token:
            login(token=args.hf_token)
        else:
            login()

        upload_folder(
            folder_path=final_dir,
            repo_id=args.hf_repo_id,
            repo_type="model",
        )
        print(f"Uploaded to Hugging Face Hub: {args.hf_repo_id}")


if __name__ == "__main__":
    main()

