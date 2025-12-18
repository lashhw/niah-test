#!/usr/bin/env python3
"""
Needle-In-A-Haystack (NIAH) test for `jiawei-ucas/ConsistentChat`.

Insert one "needle" (a secret code) into a multi-turn chat (the haystack) at a
controllable location (e.g., last 10% of total tokens), then ask a retrieval
question and check whether the model returns the expected code.
"""

import argparse
import random

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

DATASET_NAME = "jiawei-ucas/ConsistentChat"
SPLIT = "train"
NEEDLE_TEXT = "By the way, the secret code is {needle}. Remember it exactly."
QUESTION = "What is the secret code? Reply with only the code."


def _role(from_field: str) -> str:
    if from_field == "human":
        return "user"
    if from_field == "gpt":
        return "assistant"
    raise ValueError(f"Unexpected role in dataset: {from_field!r}")


def build_messages(example):
    return [
        {"role": _role(turn.get("from", "")), "content": str(turn.get("value", ""))}
        for turn in example["conversations"]
    ]


def n_tokens(tokenizer, messages) -> int:
    return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))


def find_insert_index_last_pct(tokenizer, messages, last_pct: float, min_index: int):
    if not (0.0 < last_pct <= 1.0):
        raise ValueError("--last_pct must be in (0, 1].")
    total = n_tokens(tokenizer, messages)
    target = int(total * (1.0 - last_pct))  # insert at start of last X% tokens
    for i in range(min_index, len(messages) + 1):
        if n_tokens(tokenizer, messages[:i]) >= target:
            return i, total
    return len(messages), total


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--last_pct", type=float, default=0.10)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    args = parser.parse_args(argv)

    random.seed(args.seed)

    ds = load_dataset(DATASET_NAME)[SPLIT]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
    )

    needle = str(random.randint(10**8, 10**9 - 1))
    needle_msg = {"role": "user", "content": NEEDLE_TEXT.format(needle=needle)}
    indices = random.sample(range(len(ds)), k=min(args.n, len(ds)))
    correct = 0

    for i, idx in enumerate(indices, 1):
        ex = ds[idx]
        base = build_messages(ex)
        insert_at, total_tokens = find_insert_index_last_pct(tok, base, last_pct=args.last_pct, min_index=0)

        msgs = base[:insert_at] + [needle_msg] + base[insert_at:]
        msgs.append({"role": "user", "content": QUESTION})

        input_ids = tok.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        answer = tok.decode(out[0, input_ids.shape[-1] :], skip_special_tokens=True).strip()

        ok = needle in answer
        correct += int(ok)

        frac = n_tokens(tok, base[:insert_at]) / max(1, total_tokens)
        print(f"[{i}/{len(indices)}] idx={idx} ok={ok} frac≈{frac:.3f}")
        if not ok:
            print(f"  expected={needle!r}")
            print(f"  answer={answer!r}")

    total = len(indices)
    print(f"\naccuracy: {correct}/{total} = {correct/total:.3f}")
    return 0


if __name__ == "__main__":
    main()
