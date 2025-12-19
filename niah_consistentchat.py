"""
Needle-In-A-Haystack (NIAH) test for `jiawei-ucas/ConsistentChat`.

Insert one "needle" (a secret code) into a multi-turn chat (the haystack) at a
controllable location (e.g., last 10% of total tokens), then ask a retrieval
question and check whether the model returns the expected code.
"""

import argparse
import random
from tqdm import tqdm

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

DATASET_NAME = "jiawei-ucas/ConsistentChat"
SPLIT = "train"
NEEDLE_TEXT = "By the way, the secret code for urei-dfjc is {needle}. Remember it exactly."
QUESTION = "What is the secret code for urei-dfjc? Reply with only the code."


def _role(from_field):
    if from_field == "human":
        return "user"
    if from_field == "gpt":
        return "assistant"
    raise ValueError(f"Unexpected role in dataset: {from_field!r}")


def build_messages(example):
    return [
        {"role": _role(turn["from"]), "content": str(turn["value"])}
        for turn in example["conversations"]
    ]


def n_tokens(tokenizer, messages):
    if not messages:
        return 0
    return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))


def find_insert_index_last_pct(tokenizer, messages, last_pct):
    if not (0.0 < last_pct <= 1.0):
        raise ValueError("--last_pct must be in (0, 1].")
    total = n_tokens(tokenizer, messages)
    target = int(total * (1.0 - last_pct))  # insert at start of last X% tokens
    for i in range(len(messages), -1, -1):
        if n_tokens(tokenizer, messages[:i]) < target:
            return i + 1
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total", type=int, default=1)
    parser.add_argument("--last_pct", type=float, default=0.10)
    parser.add_argument(
        "--min_tokens",
        type=int,
        default=0,
        help="If > 0, concatenate multiple conversations until the prompt reaches at least this many tokens (excluding generation).",
    )
    parser.add_argument("--max_new_tokens", type=int, default=32)
    args = parser.parse_args()

    random.seed(args.seed)

    ds = load_dataset(DATASET_NAME, split=SPLIT)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
    )

    needle = str(random.randint(10**8, 10**9 - 1))
    total = min(args.total, len(ds))
    correct = 0

    for _ in tqdm(range(total)):
        msgs = []
        while True:
            msgs.extend(build_messages(ds[random.randrange(len(ds))]))
            if not args.min_tokens or n_tokens(tok, msgs) >= args.min_tokens:
                break

        insert_at = find_insert_index_last_pct(tok, msgs, args.last_pct)
        needle_at = min(max(insert_at - 1, 0), len(msgs) - 1)
        msgs[needle_at]["content"] = msgs[needle_at]["content"] + " " + NEEDLE_TEXT.format(needle=needle)
        msgs.append({"role": "user", "content": QUESTION})

        inputs = tok.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tok.eos_token_id,
            )
        answer = tok.decode(out[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()

        ok = needle in answer
        correct += int(ok)

    print(f"\naccuracy: {correct}/{total} = {correct/total:.3f}")


if __name__ == "__main__":
    main()
