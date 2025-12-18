#!/usr/bin/env python3
"""
Needle-In-A-Haystack (NIAH) test for `jiawei-ucas/ConsistentChat`.

Insert one "needle" (a secret code) into a multi-turn chat (the haystack) at a
controllable location (e.g., last 10% of total tokens), then ask a retrieval
question and check whether the model returns the expected code.
"""

import argparse
import random
import re


DATASET_NAME = "jiawei-ucas/ConsistentChat"
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"


def _role(from_field: str) -> str:
    t = (from_field or "").strip().lower()
    if t in {"gpt", "assistant", "model"}:
        return "assistant"
    return "user"  # includes "human"


def build_messages(example, system: str | None):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    for turn in example["conversations"]:
        msgs.append({"role": _role(turn.get("from", "")), "content": str(turn.get("value", ""))})
    return msgs


def normalize(text: str) -> str:
    text = text.strip()
    return re.sub(r"^\s*[\"'`]+|[\"'`]+\s*$", "", text).strip()


def n_tokens(tokenizer, messages) -> int:
    return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))


def find_insert_index_last_pct(tokenizer, messages, last_pct: float, min_index: int):
    if not (0.0 < last_pct <= 1.0):
        raise ValueError("--last-pct must be in (0, 1].")
    total = n_tokens(tokenizer, messages)
    target = int(total * (1.0 - last_pct))  # insert at start of last X% tokens
    for i in range(min_index, len(messages) + 1):
        if n_tokens(tokenizer, messages[:i]) >= target:
            return i, total
    return len(messages), total


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--split", default="train")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--offline", action="store_true")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=1, help="How many chats to test.")
    parser.add_argument("--example-index", type=int, default=None, help="If set, test only this row index.")

    parser.add_argument("--last-pct", type=float, default=0.10, help="Insert at start of last X% tokens.")
    parser.add_argument(
        "--after",
        type=int,
        default=None,
        help="If set, insert after this message index (0-based, in the baseline chat).",
    )

    parser.add_argument("--needle", default=None, help="Secret code to insert (default: random digits).")
    parser.add_argument("--needle-role", choices=["user", "assistant"], default="user")
    parser.add_argument(
        "--needle-text",
        default="IMPORTANT: The secret code is {needle}. Remember it exactly.",
        help="Must contain {needle}.",
    )
    parser.add_argument("--question", default="What is the secret code? Reply with only the code.")
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--match", choices=["exact", "contains"], default="exact")

    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    random.seed(args.seed)

    try:
        import torch
        from datasets import DownloadConfig, load_dataset  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception as e:
        raise SystemExit(f"Missing deps (`datasets`, `transformers`, `torch`): {e}")

    download_config = DownloadConfig(local_files_only=args.offline) if args.offline else None
    dataset = load_dataset(
        DATASET_NAME,
        cache_dir=args.cache_dir,
        streaming=False,
        download_config=download_config,
    )
    if args.split not in dataset:
        raise SystemExit(f'Unknown split "{args.split}". Available: {list(dataset.keys())}')
    ds = dataset[args.split]

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=args.offline)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        local_files_only=args.offline,
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    needle = args.needle or str(random.randint(10**7, 10**8 - 1))
    needle_msg = {"role": args.needle_role, "content": args.needle_text.format(needle=needle)}

    if args.example_index is not None:
        indices = [args.example_index]
    else:
        indices = random.sample(range(len(ds)), k=min(args.n, len(ds)))
    correct = 0

    for i, idx in enumerate(indices, 1):
        ex = ds[idx]
        base = build_messages(ex, system=args.system)
        min_index = 1 if args.system else 0

        if args.after is not None:
            insert_at = max(min_index, min(len(base), args.after + 1))
            total_tokens = n_tokens(tok, base)
        else:
            insert_at, total_tokens = find_insert_index_last_pct(tok, base, last_pct=args.last_pct, min_index=min_index)

        msgs = base[:insert_at] + [needle_msg] + base[insert_at:]
        msgs.append({"role": "user", "content": args.question})

        input_ids = tok.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        do_sample = args.temperature > 0
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=args.temperature if do_sample else None,
                top_p=args.top_p if do_sample else None,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        answer = tok.decode(out[0, input_ids.shape[-1] :], skip_special_tokens=True).strip()

        got = normalize(answer)
        exp = normalize(needle)
        ok = (exp in got) if args.match == "contains" else (got == exp)
        correct += int(ok)

        frac = n_tokens(tok, base[:insert_at]) / max(1, total_tokens)
        print(f"[{i}/{len(indices)}] idx={idx} ok={ok} insert_at={insert_at} frac≈{frac:.3f}")
        if args.verbose:
            print(f"  intent={ex.get('intent','')!r}")
            print(f"  scenario={ex.get('scenario','')!r}")
            print(f"  needle={needle!r}")
            print(f"  answer={answer!r}")

    total = len(indices)
    print(f"\naccuracy: {correct}/{total} = {correct/total:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
