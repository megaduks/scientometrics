"""Re-rank, paginate and cite.

Each hypothetical paper (one per query) takes its search results, re-ranks them by a weighted
score, splits them into pages, then cites papers from the first pages. Queries are processed
in order of their year and in random order within a year. Without a time constraint all
queries share one year, so the order is fully random. The citation-count feature is either
``static`` (the papers' real citation counts, as a search engine shows them today) or
``dynamic`` (citations received so far from hypothetical papers in this run, starting at zero).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

FEATURES = ("similarity", "recency", "citations", "venue")


@dataclass(frozen=True)
class SimConfig:
    weights: tuple[float, float, float, float] = (0.5, 0.3, 0.1, 0.1)  # similarity, recency, citations, venue
    page_size: int = 10
    # Per-page mode: cite exactly per_page[i] distinct papers chosen uniformly from page i+1.
    per_page: tuple[int, ...] | None = (3, 1, 1)
    # Geometric mode (per_page=None): each paper cites k distinct papers. A result on page p
    # (0-based) is chosen with weight exp(-decay * p). k=None uses the real number of
    # in-corpus references of the paper this query stands in for.
    k: int | None = None
    decay: float = 1.0
    citation_signal: str = "dynamic"   # "dynamic" (accumulated in this run) | "static" (real internal counts)
    seed: int = 0

    def __post_init__(self):
        object.__setattr__(self, "weights", tuple(float(w) for w in self.weights))
        if self.per_page is not None:
            object.__setattr__(self, "per_page", tuple(int(c) for c in self.per_page))
        if len(self.weights) != 4:
            raise ValueError("weights must have 4 entries: " + ", ".join(FEATURES))
        if self.citation_signal not in ("dynamic", "static"):
            raise ValueError("citation_signal must be 'dynamic' or 'static'")

    @property
    def run_id(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def simulate(cfg: SimConfig, cand_idx: np.ndarray, cand_sim: np.ndarray, query_year: np.ndarray,
             query_k: np.ndarray, doc_year: np.ndarray, doc_venue: np.ndarray,
             doc_static_citations: np.ndarray) -> np.ndarray:
    """Return simulated citation counts for every document, shape ``(n_docs,)``."""
    rng = np.random.default_rng(cfg.seed)
    n, top_m = cand_idx.shape
    w = np.asarray(cfg.weights, np.float32)
    year = doc_year.astype(np.float32)
    venue = doc_venue.astype(np.float32)
    static_cit = np.log1p(doc_static_citations).astype(np.float32)
    counts = np.zeros(len(doc_year), np.int64)
    page_of_rank = np.arange(top_m) // cfg.page_size
    rank_weight = np.exp(-cfg.decay * page_of_rank)

    order = np.lexsort((rng.random(n), query_year))       # chronological, random within a year
    for q in order:
        c = np.asarray(cand_idx[q])
        nv = int(np.count_nonzero(c >= 0))                 # valid candidates come first
        if nv == 0:
            continue
        c = c[:nv]
        cit = np.log1p(counts[c]).astype(np.float32) if cfg.citation_signal == "dynamic" else static_cit[c]
        score = (w[0] * _minmax(np.asarray(cand_sim[q, :nv], np.float32))
                 + w[1] * _minmax(year[c])
                 + w[2] * _minmax(cit)
                 + w[3] * _minmax(venue[c]))
        ranked = c[np.argsort(-score, kind="stable")]

        if cfg.per_page is not None:
            picks = []
            for p, kp in enumerate(cfg.per_page):
                page = ranked[p * cfg.page_size:(p + 1) * cfg.page_size]
                if len(page) == 0:
                    break
                if kp > 0:
                    picks.append(rng.choice(page, min(kp, len(page)), replace=False))
            if not picks:
                continue
            cited = np.concatenate(picks)
        else:
            k = int(cfg.k if cfg.k is not None else query_k[q])
            k = min(k, nv)
            if k == 0:
                continue
            p = rank_weight[:nv] / rank_weight[:nv].sum()
            cited = ranked[rng.choice(nv, k, replace=False, p=p)]
        counts[cited] += 1                                 # cited is duplicate-free
    return counts
