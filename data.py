"""
this file loads the dataset and tokenizes the data so the the model can use it, streaming and casual LM chunking

Sources:
- Hugging Face Datasets:
  https://huggingface.co/docs/datasets/loading
  https://huggingface.co/docs/datasets/stream

- DCLM Baseline dataset (ML Foundations):
  https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0

- nanoGPT (Karpathy):
  contiguous token buffer + chunking into (block_size + 1)
  https://github.com/karpathy/nanoGPT

- GPT-style causal language modeling:
  Brown et al. (GPT-3), Radford et al. (GPT-2)
  next-token prediction with shifted targets

- Uses GPT-2 tokenizer (vocab_size = 50257)
"""
import math
import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


class DCLMTokenStream(IterableDataset):
    """
    Streams mlfoundations/dclm-baseline-1.0 and yields causal LM examples:
      idx:     [block_size]
      targets: [block_size]
    Stops after approximately target_tokens 
    """

    def __init__(
        self,
        tokenizer_name: str = "gpt2",
        split: str = "train",
        block_size: int = 1024,
        target_tokens: int = 5_000_000_000,
        shuffle_buffer: int | None = None,
        seed: int = 42,
        add_eos_between_docs: bool = True,
    ):
        self.tokenizer_name = tokenizer_name
        self.split = split
        self.block_size = block_size
        self.target_tokens = target_tokens
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.add_eos_between_docs = add_eos_between_docs

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.eos_token_id is None:
            raise ValueError(f"{tokenizer_name} tokenizer has no eos_token_id")

    def _get_stream(self):
        ds = load_dataset(
            "mlfoundations/dclm-baseline-1.0",
            split=self.split,
            streaming=True,
        )
        if self.shuffle_buffer and self.shuffle_buffer > 0:
            ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=self.seed)
        return ds

    def __iter__(self):
        ds = self._get_stream()
        buffer: list[int] = []
        tokens_seen = 0
        emitted = 0

        for example in ds:
            text = example.get("text", None)
            if not text:
                continue

            ids = self.tokenizer.encode(text, add_special_tokens=False)

          
            if self.add_eos_between_docs:
                ids.append(self.tokenizer.eos_token_id)

            tokens_seen += len(ids)
            buffer.extend(ids)

            # Emit as many contiguous training chunks as possible
            chunk_len = self.block_size + 1
            while len(buffer) >= chunk_len:
                chunk = buffer[:chunk_len]
                del buffer[:chunk_len]

                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)

                emitted += 1
                yield {"idx": x, "targets": y}

            if tokens_seen >= self.target_tokens:
                break

        # leftover tokens are dropped on purpose for fixed-length batching


def build_train_dataloader(
    tokenizer_name: str = "gpt2",
    block_size: int = 1024,
    batch_size: int = 4,
    target_tokens: int = 5_000_000_000,
    shuffle_buffer: int | None = 10_000,
    num_workers: int = 0,
    pin_memory: bool = True,
    seed: int = 42,
):
    if num_workers != 0:
        raise ValueError("num_workers must be 0 for this streaming IterableDataset.")

    ds = DCLMTokenStream(
        tokenizer_name=tokenizer_name,
        split="train",
        block_size=block_size,
        target_tokens=target_tokens,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

def estimate_num_steps(
    target_tokens: int,
    block_size: int,
    batch_size: int,
    grad_accum_steps: int = 1,
) -> int:
    tokens_per_step = block_size * batch_size * grad_accum_steps
    return math.ceil(target_tokens / tokens_per_step)

