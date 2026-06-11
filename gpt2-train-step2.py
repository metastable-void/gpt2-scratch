#!/usr/bin/env python3
"""
Step 2: fine-tune the from-scratch GPT-2 base model on a small corpus of
short free-form texts (e.g. the clinical dataset), one item per JSONL line.

Key differences from pretraining:
- UNPACKED: one item per sequence, dynamically padded per batch.
- Label masking is based on attention_mask, NOT pad_token_id. With pad == eos,
  masking by token id would silently remove every EOS from the loss
  (the bug we hit in pretraining). Here: labels = input_ids, with -100 only
  where attention_mask == 0. The real, final EOS has attention_mask == 1
  and stays in the loss.
- Low LR, few epochs, eval per epoch with best-checkpoint selection.
- Optional memorization audit: generate samples and measure verbatim
  token n-gram overlap against the training set.

Usage:

  # fine-tune (base model from HF Hub or a local path)
  python finetune_gpt2_short_texts.py \
      --base_model your-org/gpt2-ja-base \
      --data_files ./clinical.jsonl --text_column text \
      --output_dir ./gpt2-ja-clinical --bf16

  # fine-tune + audit with 2000 generated samples
  python finetune_gpt2_short_texts.py ... --audit_samples 2000

  # audit an already fine-tuned model without retraining
  python finetune_gpt2_short_texts.py \
      --base_model ./gpt2-ja-clinical \
      --data_files ./clinical.jsonl --text_column text \
      --output_dir ./gpt2-ja-clinical --audit_samples 2000 --audit_only
"""

import argparse
import json
import math
import os
import statistics

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune GPT-2 base on short texts (unpacked).")
    # Model
    p.add_argument("--base_model", required=True,
                   help="HF Hub repo id (e.g. your-org/gpt2-ja-base) or local path. "
                        "Private repos: set the HF_TOKEN environment variable.")
    # Data
    p.add_argument("--dataset", default="json", help="HF dataset name, or 'json' for local files")
    p.add_argument("--dataset_config", default=None)
    p.add_argument("--data_files", default=None, help="Path/glob to JSONL file(s)")
    p.add_argument("--split", default="train")
    p.add_argument("--text_column", required=True)
    p.add_argument("--validation_fraction", type=float, default=0.05)
    p.add_argument("--max_length", type=int, default=256,
                   help="Max tokens per item incl. EOS; longer items are truncated")
    # Training
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_train_epochs", type=float, default=2.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=32)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_proc", type=int, default=4)
    # Memorization audit
    p.add_argument("--audit_samples", type=int, default=0,
                   help="If > 0, generate this many unconditional samples after "
                        "training and measure n-gram overlap with the training set")
    p.add_argument("--audit_only", action="store_true",
                   help="Skip training; load --base_model as the model to audit")
    p.add_argument("--audit_ngram", type=int, default=13,
                   help="Token n-gram size for the verbatim-overlap check")
    p.add_argument("--gen_max_new_tokens", type=int, default=256)
    p.add_argument("--gen_temperature", type=float, default=1.0)
    p.add_argument("--gen_top_p", type=float, default=0.95)
    p.add_argument("--gen_batch_size", type=int, default=64)
    return p.parse_args()


def load_items(args):
    kwargs = {}
    if args.dataset_config:
        kwargs["name"] = args.dataset_config
    if args.data_files:
        kwargs["data_files"] = args.data_files
    ds = load_dataset(args.dataset, split=args.split, **kwargs)
    drop = [c for c in ds.column_names if c != args.text_column]
    if drop:
        ds = ds.remove_columns(drop)
    ds = ds.filter(
        lambda batch: [t is not None and t.strip() != "" for t in batch[args.text_column]],
        batched=True, num_proc=args.num_proc, desc="Dropping empty items",
    )
    return ds


def tokenize_items(ds, tok, text_column, max_length, num_proc):
    """One item per sequence, EOS appended, no padding here (done per batch)."""
    eos_id = tok.eos_token_id

    def fn(batch):
        out = tok(batch[text_column], truncation=True, max_length=max_length - 1)
        out["input_ids"] = [ids + [eos_id] for ids in out["input_ids"]]
        out["attention_mask"] = [m + [1] for m in out["attention_mask"]]
        return out

    return ds.map(fn, batched=True, num_proc=num_proc,
                  remove_columns=ds.column_names, desc="Tokenizing")


def make_collator(tok):
    """Dynamic padding; labels masked by ATTENTION MASK, never by token id.

    pad == eos in our setup, so masking labels where token == pad_token_id
    would erase every EOS from the loss. attention_mask distinguishes the
    real final EOS (mask 1) from padding (mask 0)."""

    def collator(features):
        batch = tok.pad(features, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch

    return collator


def run_training(args, tok, model, train_ds, eval_ds):
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        ddp_find_unused_parameters=False,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=make_collator(tok),
    )
    trainer.train()

    metrics = trainer.evaluate()
    print(f"Final eval loss: {metrics['eval_loss']:.4f} "
          f"(ppl {math.exp(metrics['eval_loss']):.1f})")

    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    model.generation_config.eos_token_id = tok.eos_token_id
    model.generation_config.pad_token_id = tok.pad_token_id
    model.generation_config.save_pretrained(args.output_dir)
    return trainer.model


@torch.no_grad()
def memorization_audit(args, tok, model, train_token_ids):
    """Generate unconditional samples; report verbatim n-gram overlap with training data."""
    n = args.audit_ngram
    device = next(model.parameters()).device
    model.eval()

    # Build the set of all training n-grams (tuples of token ids).
    train_ngrams = set()
    for ids in train_token_ids:
        for i in range(len(ids) - n + 1):
            train_ngrams.add(tuple(ids[i : i + n]))
    print(f"[audit] {len(train_ngrams):,} unique {n}-gram(s) in training data")

    gen_cfg = GenerationConfig(
        do_sample=True,
        temperature=args.gen_temperature,
        top_p=args.gen_top_p,
        max_new_tokens=args.gen_max_new_tokens,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )

    samples, flagged, lengths = [], 0, []
    bs = args.gen_batch_size
    remaining = args.audit_samples
    while remaining > 0:
        cur = min(bs, remaining)
        inp = torch.full((cur, 1), tok.eos_token_id, dtype=torch.long, device=device)
        out = model.generate(inp, attention_mask=torch.ones_like(inp),
                             generation_config=gen_cfg)
        for row in out:
            ids = row[1:].tolist()  # drop the EOS used as BOS
            if tok.eos_token_id in ids:
                ids = ids[: ids.index(tok.eos_token_id)]
                terminated = True
            else:
                terminated = False
            lengths.append(len(ids))
            # longest run of consecutive overlapping n-grams -> longest verbatim span
            hit, longest, run = False, 0, 0
            for i in range(len(ids) - n + 1):
                if tuple(ids[i : i + n]) in train_ngrams:
                    hit = True
                    run += 1
                    longest = max(longest, run + n - 1)
                else:
                    run = 0
            flagged += int(hit)
            samples.append({
                "text": tok.decode(ids, clean_up_tokenization_spaces=False),
                "n_tokens": len(ids),
                "terminated": terminated,
                "verbatim_hit": hit,
                "longest_verbatim_span_tokens": longest,
            })
        remaining -= cur
        print(f"[audit] generated {len(samples)}/{args.audit_samples}", end="\r")
    print()

    out_path = os.path.join(args.output_dir, "audit_samples.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    terminated_frac = sum(s["terminated"] for s in samples) / len(samples)
    print(f"[audit] samples written to {out_path}")
    print(f"[audit] terminated before cap: {terminated_frac:.1%}")
    print(f"[audit] length: mean {statistics.mean(lengths):.1f}, "
          f"median {statistics.median(lengths)}, max {max(lengths)}")
    print(f"[audit] samples containing a verbatim training {n}-gram: "
          f"{flagged}/{len(samples)} ({flagged / len(samples):.2%})")
    spans = [s["longest_verbatim_span_tokens"] for s in samples if s["verbatim_hit"]]
    if spans:
        print(f"[audit] longest verbatim span: {max(spans)} tokens "
              f"(inspect those samples in the JSONL)")


def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"Loading model and tokenizer from {args.base_model} ...")
    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print("Loading items...")
    raw = load_items(args)
    print(f"  {len(raw):,} items")

    tokenized = tokenize_items(raw, tok, args.text_column, args.max_length, args.num_proc)
    split = tokenized.train_test_split(test_size=args.validation_fraction, seed=args.seed)
    train_ds, eval_ds = split["train"], split["test"]
    n_tokens = sum(len(x) for x in train_ds["input_ids"])
    print(f"  train {len(train_ds):,} / eval {len(eval_ds):,} items, "
          f"~{n_tokens / 1e6:.1f}M train tokens")

    if not args.audit_only:
        model = run_training(args, tok, model, train_ds, eval_ds)

    if args.audit_samples > 0:
        memorization_audit(args, tok, model, train_ds["input_ids"])


if __name__ == "__main__":
    main()