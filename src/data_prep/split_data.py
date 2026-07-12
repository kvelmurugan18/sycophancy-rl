"""Split the merged episode file into train/test JSONL files.

A single shuffle followed by a deterministic 80/20 cutoff produces the
two splits. Splitting is a deliberate upstream step so that no episode
appearing in ``data/splits/test.jsonl`` is ever seen during training —
this prevents *data leakage*, where a model could otherwise memorize
an evaluation prompt at training time and inflate its eval scores
without actually learning to resist pushback.

Randomness is seeded at the top of :func:`main` so reruns produce
byte-identical splits.
"""

import json
import random
from pathlib import Path

PROCESSED_FILE = Path("data/processed/merged_episodes.jsonl")
SPLITS_DIR = Path("data/splits")

_TEST_FRACTION = 0.2


def main() -> None:
    """Shuffle the merged episodes and write 80/20 train/test JSONL files."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(42)

    with PROCESSED_FILE.open("r", encoding="utf-8") as f:
        episodes = [json.loads(line) for line in f if line.strip()]

    random.shuffle(episodes)

    # Guarantee at least one record lands in each split, even for the
    # tiny mock dataset (4 episodes → would otherwise round to 0 test).
    n_test = max(1, int(len(episodes) * _TEST_FRACTION))
    n_test = min(n_test, len(episodes) - 1)  # leave at least 1 for train
    n_train = len(episodes) - n_test

    train_episodes = episodes[:n_train]
    test_episodes = episodes[n_train:]

    train_path = SPLITS_DIR / "train.jsonl"
    test_path = SPLITS_DIR / "test.jsonl"
    for path, rows in ((train_path, train_episodes), (test_path, test_episodes)):
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Split complete: {n_train} train, {n_test} test episodes.")


if __name__ == "__main__":
    main()
