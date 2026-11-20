"""Corpus loading and empirical citation counts."""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# AMiner DBLP v10 stores 50 in `n_citation` when the true count is unknown.
NCITATION_PLACEHOLDER = 50


@dataclass
class Corpus:
    ids: np.ndarray                 # (m,) str, AMiner paper ids
    titles: np.ndarray              # (m,) str
    year: np.ndarray                # (m,) int16
    gov_score: np.ndarray           # (m,) float32, ministry venue points
    n_citation: np.ndarray          # (m,) int64, AMiner global count (50 = unknown)
    internal_citations: np.ndarray  # (m,) int64, times cited by other corpus papers
    internal_refs: np.ndarray       # (m,) int64, references pointing into the corpus

    def __len__(self) -> int:
        return len(self.ids)


def _count_internal(ids: pd.Series, references: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    pos = {pid: k for k, pid in enumerate(ids)}
    cited = np.zeros(len(ids), np.int64)
    refs_in = np.zeros(len(ids), np.int64)
    for k, raw in enumerate(references):
        if not isinstance(raw, str) or raw == "[]":
            continue
        targets = {pos[r] for r in ast.literal_eval(raw) if r in pos and pos[r] != k}
        refs_in[k] = len(targets)
        for j in targets:
            cited[j] += 1
    return cited, refs_in


def load_corpus(csv_path: str | Path, cache_dir: str | Path) -> Corpus:
    """Load ``articles_with_score_df.csv``; cache the parsed arrays as ``corpus.npz``."""
    cache = Path(cache_dir) / "corpus.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return Corpus(**{k: z[k] for k in z.files})
    df = pd.read_csv(csv_path, usecols=["id", "title", "year", "references", "n_citation", "gov_score"])
    cited, refs_in = _count_internal(df["id"], df["references"])
    corpus = Corpus(
        ids=df["id"].to_numpy(str),
        titles=df["title"].fillna("").to_numpy(str),
        year=df["year"].to_numpy(np.int16),
        gov_score=df["gov_score"].to_numpy(np.float32),
        n_citation=df["n_citation"].to_numpy(np.int64),
        internal_citations=cited,
        internal_refs=refs_in,
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **corpus.__dict__)
    return corpus


def distribution(counts: np.ndarray) -> dict[int, int]:
    """``{number_of_citations: number_of_papers}``, zeros included."""
    values, freq = np.unique(counts, return_counts=True)
    return {int(v): int(f) for v, f in zip(values, freq)}
