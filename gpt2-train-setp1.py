#!/usr/bin/env python3
"""
Train an *empty* (randomly initialized) GPT-2 from scratch with HF transformers,
including custom byte-level BPE tokenizer training on the same dataset.

- Simple EOS setup: a single special token <|endoftext|> used as eos/bos/pad.
- Each document gets EOS appended, then everything is packed into fixed-size blocks.

Usage examples:

  # HF Hub dataset
  python train_gpt2_scratch.py \
      --dataset wikimedia/wikipedia --dataset_config 20231101.ja \
      --text_column text --output_dir ./out

  # Local JSONL (one item per line, text in the "text" field)
  python train_gpt2_scratch.py \
      --dataset json --data_files ./corpus.jsonl \
      --text_column text --output_dir ./out
"""

import argparse
import math

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

EOS = "<|endoftext|>"


def parse_args():
    p = argparse.ArgumentParser(description="Train GPT-2 from scratch with a custom tokenizer.")
    # Data
    p.add_argument("--dataset", required=True, help="HF dataset name, or 'json'/'csv' for local files")
    p.add_argument("--dataset_config", default=None, help="Dataset config name (if any)")
    p.add_argument("--data_files", default=None, help="Path/glob for local files (with --dataset json)")
    p.add_argument("--split", default="train")
    p.add_argument("--text_column", required=True, help="Name of the text column to train on")
    p.add_argument("--validation_fraction", type=float, default=0.01)
    # Tokenizer
    p.add_argument("--vocab_size", type=int, default=32768)
    p.add_argument("--tokenizer_sample_size", type=int, default=200_000,
                   help="Max number of items used to train the tokenizer (0 = all)")
    # Model
    p.add_argument("--block_size", type=int, default=512, help="Context length (n_positions)")
    p.add_argument("--n_layer", type=int, default=12)
    p.add_argument("--n_head", type=int, default=12)
    p.add_argument("--n_embd", type=int, default=768)
    # Training
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=-1, help="Overrides epochs if > 0")
    p.add_argument("--per_device_train_batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=6e-4)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_proc", type=int, default=4, help="Workers for dataset map()")
    return p.parse_args()


def load_raw_dataset(args):
    kwargs = {}
    if args.dataset_config:
        kwargs["name"] = args.dataset_config
    if args.data_files:
        kwargs["data_files"] = args.data_files
    ds = load_dataset(args.dataset, split=args.split, **kwargs)
    # Keep only the text column; drop everything else early.
    drop = [c for c in ds.column_names if c != args.text_column]
    if drop:
        ds = ds.remove_columns(drop)
    return ds


def train_tokenizer(ds, text_column, vocab_size, sample_size):
    """Train a byte-level BPE tokenizer (GPT-2 style) from scratch on the dataset."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOS],  # gets ID 0; never split by BPE
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # full byte coverage, no UNK
    )

    n = len(ds) if sample_size in (0, None) else min(sample_size, len(ds))
    sample = ds.shuffle(seed=0).select(range(n))

    def batch_iter(batch_size=1000):
        for i in range(0, len(sample), batch_size):
            yield sample[i : i + batch_size][text_column]

    tokenizer.train_from_iterator(batch_iter(), trainer=trainer, length=n)

    # Simple EOS setup: one token plays eos/bos/pad. Fine for packed causal LM
    # training, where the attention mask is all-ones and pad is never used.
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token=EOS,
        bos_token=EOS,
        pad_token=EOS,
    )
    return hf_tok


def tokenize_and_pack(ds, tok, text_column, block_size, num_proc):
    """Tokenize each item, append EOS, concatenate, and chunk into block_size pieces."""
    eos_id = tok.eos_token_id

    def tokenize_fn(batch):
        out = tok(batch[text_column])
        out["input_ids"] = [ids + [eos_id] for ids in out["input_ids"]]
        return {"input_ids": out["input_ids"]}

    ds = ds.map(
        tokenize_fn,
        batched=True,
        num_proc=num_proc,
        remove_columns=ds.column_names,
        desc="Tokenizing",
    )

    def pack_fn(batch):
        concat = [t for ids in batch["input_ids"] for t in ids]
        total = (len(concat) // block_size) * block_size
        blocks = [concat[i : i + block_size] for i in range(0, total, block_size)]
        return {"input_ids": blocks}
        # Leftover tail (< block_size) of each map-batch is dropped; negligible loss.

    ds = ds.map(pack_fn, batched=True, num_proc=num_proc, desc="Packing")
    return ds


def build_model(tok, args):
    config = GPT2Config(
        vocab_size=len(tok),
        n_positions=args.block_size,
        n_ctx=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    model = GPT2LMHeadModel(config)  # random init — NOT from_pretrained
    print(f"Model parameters: {model.num_parameters() / 1e6:.1f}M")
    return model


def main():
    args = parse_args()

    print("Loading dataset...")
    raw = load_raw_dataset(args)
    print(f"  {len(raw):,} items")

    print("Training tokenizer...")
    tok = train_tokenizer(raw, args.text_column, args.vocab_size, args.tokenizer_sample_size)
    tok.save_pretrained(args.output_dir)
    print(f"  vocab size: {len(tok):,} (saved to {args.output_dir})")

    print("Tokenizing and packing...")
    lm_ds = tokenize_and_pack(raw, tok, args.text_column, args.block_size, args.num_proc)
    n_tokens = len(lm_ds) * args.block_size
    print(f"  {len(lm_ds):,} blocks of {args.block_size} = {n_tokens / 1e6:.1f}M tokens")

    split = lm_ds.train_test_split(test_size=args.validation_fraction, seed=args.seed)
    train_ds, eval_ds = split["train"], split["test"]

    model = build_model(tok, args)

    # mlm=False -> causal LM: labels are input_ids shifted internally by the model.
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        adam_beta2=0.95,
        bf16=args.bf16,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    trainer.train()

    metrics = trainer.evaluate()
    print(f"Final eval loss: {metrics['eval_loss']:.4f} "
          f"(ppl {math.exp(metrics['eval_loss']):.1f})")

    trainer.save_model(args.output_dir)  # model + config
    tok.save_pretrained(args.output_dir)
    model.generation_config.eos_token_id = tok.eos_token_id
    model.generation_config.pad_token_id = tok.pad_token_id
    model.generation_config.save_pretrained(args.output_dir)
    print(f"Done. Everything saved to {args.output_dir}")


if __name__ == "__main__":
    main()