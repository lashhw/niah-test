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
python explore_consistentchat.py --print-dataset --show-features --show-stats
python explore_consistentchat.py --split train --head 3
python explore_consistentchat.py --split train --example-index 0
```

If you want to avoid any network access, use:

```bash
python explore_consistentchat.py --offline --print-dataset --show-features --show-stats
```

## Needle-In-A-Haystack (NIAH) test

`niah_consistentchat.py` inserts a "needle" (a secret code) into the chat, near a controllable location (by token position), then appends a retrieval question and checks whether the model returns the expected code.

Examples:

```bash
# Insert needle at the start of the last 10% of the chat (by tokens), then query at the end
python niah_consistentchat.py --last-pct 0.10 --example-index 0 --verbose

# Place needle near the very end (last 1%)
python niah_consistentchat.py --last-pct 0.01 --example-index 0 --verbose

# Place needle after a specific message index (0-based, within the baseline chat)
python niah_consistentchat.py --after 12 --example-index 0 --verbose

# Run on N random chats
python niah_consistentchat.py --n 50
```
