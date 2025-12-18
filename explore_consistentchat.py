#!/usr/bin/env python3
"""
Explore the `jiawei-ucas/ConsistentChat` Hugging Face dataset.

Example:
  python explore_consistentchat.py --print-dataset --show-features --show-stats
  python explore_consistentchat.py --split train --head 3
  python explore_consistentchat.py --split train --example-index 0

Notes:
  - This script does not use streaming.
  - If you already have the dataset cached, it will load locally.
  - For strict offline mode, pass `--offline`.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pprint import pformat, pprint
from typing import Any, Iterable


DATASET_NAME = "jiawei-ucas/ConsistentChat"


def _safe_len(x: Any) -> int | None:
    try:
        return len(x)  # type: ignore[arg-type]
    except Exception:
        return None


def _is_messages_like(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    first = value[0]
    if not isinstance(first, dict):
        return False
    keys = {str(k).lower() for k in first.keys()}
    return ("role" in keys and ("content" in keys or "text" in keys)) or (
        "from" in keys and ("value" in keys or "text" in keys)
    )


def _guess_conversation_field(example: dict[str, Any]) -> str | None:
    preferred = [
        "messages",
        "conversation",
        "conversations",
        "dialogue",
        "dialog",
        "chat",
        "turns",
    ]
    lowered = {k.lower(): k for k in example.keys()}
    for name in preferred:
        if name in lowered and _is_messages_like(example[lowered[name]]):
            return lowered[name]
    for key, value in example.items():
        if _is_messages_like(value):
            return key
    return None


def _truncate_repr(value: Any, max_chars: int = 500) -> str:
    text = pformat(value, width=100)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


@dataclass(frozen=True)
class SplitStats:
    num_rows: int
    keys: list[str]
    missing_counts: dict[str, int]
    type_counts: dict[str, dict[str, int]]
    conversation_field: str | None
    conversation_turns: dict[str, float] | None


def compute_split_stats(ds: Any, max_scan: int | None = None) -> SplitStats:
    num_rows = int(getattr(ds, "num_rows", len(ds)))
    scan_n = min(num_rows, max_scan) if max_scan else num_rows

    keys_union: set[str] = set()
    missing_counts: Counter[str] = Counter()
    type_counts: dict[str, Counter[str]] = defaultdict(Counter)

    first_example = ds[0] if num_rows else {}
    conversation_field = _guess_conversation_field(first_example) if first_example else None

    # Conversation metrics
    turns_list: list[int] = []

    for i in range(scan_n):
        ex = ds[i]
        if not isinstance(ex, dict):
            continue

        keys_union.update(ex.keys())
        for k, v in ex.items():
            type_counts[k][type(v).__name__] += 1

        # Missing: count keys absent in this example (requires known universe).
        # We approximate missing counts in a second pass by only considering keys seen so far.
        for k in keys_union:
            if k not in ex:
                missing_counts[k] += 1

        if conversation_field and conversation_field in ex:
            conv = ex.get(conversation_field)
            if isinstance(conv, list):
                turns_list.append(len(conv))

    conversation_turns = None
    if turns_list:
        conversation_turns = {
            "min": float(min(turns_list)),
            "max": float(max(turns_list)),
            "avg": float(sum(turns_list) / len(turns_list)),
        }

    return SplitStats(
        num_rows=num_rows,
        keys=sorted(keys_union),
        missing_counts=dict(missing_counts),
        type_counts={k: dict(c) for k, c in type_counts.items()},
        conversation_field=conversation_field,
        conversation_turns=conversation_turns,
    )


def print_head(ds: Any, n: int) -> None:
    n = max(0, n)
    for i in range(min(n, len(ds))):
        print(f"\n--- {i} ---")
        pprint(ds[i])


def print_example(ds: Any, index: int) -> None:
    if index < 0 or index >= len(ds):
        raise SystemExit(f"example-index {index} out of range (0..{len(ds) - 1})")
    pprint(ds[index])


def save_samples_jsonl(ds: Any, out_path: str, n: int) -> None:
    n = max(0, n)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(min(n, len(ds))):
            json.dump(ds[i], f, ensure_ascii=False)
            f.write("\n")


def _load_dataset(cache_dir: str | None, offline: bool) -> Any:
    try:
        from datasets import DownloadConfig, load_dataset  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependency `datasets`. Install with:\n"
            "  python -m pip install -U datasets\n\n"
            f"Import error: {e}"
        )

    download_config = None
    if offline:
        download_config = DownloadConfig(local_files_only=True)

    return load_dataset(
        DATASET_NAME,
        cache_dir=cache_dir,
        streaming=False,
        download_config=download_config,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=None, help="Optional HF datasets cache dir.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force local-only loading (sets HF_*_OFFLINE and uses local_files_only).",
    )
    parser.add_argument(
        "--split",
        default="all",
        help='Split name to explore (e.g. "train") or "all".',
    )
    parser.add_argument("--print-dataset", action="store_true", help="Print DatasetDict summary.")
    parser.add_argument("--show-features", action="store_true", help="Print split features/schema.")
    parser.add_argument("--show-stats", action="store_true", help="Compute basic stats per split.")
    parser.add_argument(
        "--max-scan",
        type=int,
        default=None,
        help="Limit examples scanned for stats (default: scan all rows).",
    )
    parser.add_argument("--head", type=int, default=0, help="Print first N examples from the split.")
    parser.add_argument(
        "--example-index",
        type=int,
        default=None,
        help="Print one example by index from the split.",
    )
    parser.add_argument(
        "--save-jsonl",
        default=None,
        help="Save first N examples to a JSONL file (requires --head N > 0).",
    )
    args = parser.parse_args(argv)

    dataset = _load_dataset(cache_dir=args.cache_dir, offline=args.offline)

    if args.print_dataset:
        print(dataset)

    if args.split == "all":
        split_names = list(dataset.keys())
    else:
        if args.split not in dataset:
            raise SystemExit(f'Unknown split "{args.split}". Available: {list(dataset.keys())}')
        split_names = [args.split]

    for split_name in split_names:
        ds = dataset[split_name]
        print(f"\n== Split: {split_name} ==")
        print(f"rows: {len(ds)}")

        if args.show_features:
            features = getattr(ds, "features", None)
            print("\nfeatures:")
            print(_truncate_repr(features))

        if args.show_stats:
            stats = compute_split_stats(ds, max_scan=args.max_scan)
            print("\nkeys:")
            print(stats.keys)
            print("\nmissing_counts (approx, within scan window):")
            print(stats.missing_counts)
            print("\ntype_counts:")
            print(stats.type_counts)
            print("\nconversation_field:")
            print(stats.conversation_field)
            if stats.conversation_turns:
                print("\nconversation_turns:")
                print(stats.conversation_turns)

        if args.example_index is not None:
            print("\nexample:")
            print_example(ds, args.example_index)

        if args.head:
            print("\nhead:")
            print_head(ds, args.head)

        if args.save_jsonl:
            if not args.head or args.head <= 0:
                raise SystemExit("--save-jsonl requires --head N where N > 0")
            save_samples_jsonl(ds, args.save_jsonl, args.head)
            print(f'\nsaved: "{args.save_jsonl}"')

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
