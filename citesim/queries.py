"""Keyword-query generation from the word lists in ``data/raw``."""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Verbs whose final consonant doubles before "-ing".
_DOUBLE_FINAL = {"Map", "Remap", "Log", "Debug", "Program", "Input", "Output", "Format", "Control"}

DEFAULT_TEMPLATE = "{verb} {adjective} {noun} {agent}"   # e.g. "Optimizing Distributed Query Scheduler"
SLOT_FILES = {"verb": "verbs.csv", "adjective": "adjectives.csv", "noun": "nouns.csv", "agent": "participles.csv"}


def gerund(verb: str) -> str:
    if verb in _DOUBLE_FINAL:
        return verb + verb[-1] + "ing"
    if verb.endswith("ie"):
        return verb[:-2] + "ying"
    if verb.endswith("e") and not verb.endswith(("ee", "ye", "oe")):
        return verb[:-1] + "ing"
    return verb + "ing"


def load_word_lists(raw_dir: str | Path) -> dict[str, list[str]]:
    lists = {}
    for slot, fname in SLOT_FILES.items():
        words = [w.strip() for w in Path(raw_dir, fname).read_text().splitlines() if w.strip()]
        lists[slot] = sorted(set(words))
    lists["verb"] = [gerund(v) for v in lists["verb"]]
    return lists


def generate_queries(n: int, lists: dict[str, list[str]], rng: np.random.Generator,
                     template: str = DEFAULT_TEMPLATE) -> list[str]:
    """``n`` queries, each slot filled independently and uniformly from its word list."""
    picks = {slot: rng.integers(0, len(words), n) for slot, words in lists.items()}
    return [template.format(**{slot: lists[slot][picks[slot][i]] for slot in lists}) for i in range(n)]
