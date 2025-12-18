# ConsistentChat exploration

This workspace contains a small script to inspect the Hugging Face dataset `jiawei-ucas/ConsistentChat`.

## Minimal snippet (from the prompt)

```python
from datasets import load_dataset

dataset = load_dataset("jiawei-ucas/ConsistentChat")
print(dataset)
print(dataset["train"][0])
```

## Scripted exploration

Run:

```bash
./venv/bin/python explore_consistentchat.py --print-dataset --show-features --show-stats
./venv/bin/python explore_consistentchat.py --split train --head 3
./venv/bin/python explore_consistentchat.py --split train --example-index 0
```

The dataset is cached locally in this environment, so it should load without extra flags.

## Needle-In-A-Haystack (NIAH) test

`niah_consistentchat.py` inserts a "needle" (a secret code) into the `train` split chat near the end of the context (by token position), then appends a retrieval question and checks exact-match accuracy using `meta-llama/Llama-3.2-1B-Instruct`.

Examples:

```bash
# Default (tests 1 random chat; inserts at last 10% by tokens)
./venv/bin/python niah_consistentchat.py

# Override the model
./venv/bin/python niah_consistentchat.py --model meta-llama/Llama-3.2-1B-Instruct

# Place needle near the very end (last 1% by tokens)
./venv/bin/python niah_consistentchat.py --last_pct 0.01

# Run on N random chats (deterministic with --seed)
./venv/bin/python niah_consistentchat.py --n 50 --seed 0
```
