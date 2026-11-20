"""Exact top-M semantic search, optionally with a publication-time constraint.

A query of year ``y`` retrieves only papers published in year ``<= y``. To search the whole
database (a new paper citing any existing paper), give every query the maximum year.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm


def retrieve(query_emb: np.ndarray, query_year: np.ndarray, doc_emb: np.ndarray, doc_year: np.ndarray,
             top_m: int, exclude: np.ndarray | None = None, device: str | None = None, batch_size: int = 128,
             out_dir: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(cand_idx, cand_sim)``, both of shape ``(n_queries, top_m)``.

    Row ``i`` holds query ``i``'s candidates sorted by decreasing cosine similarity.
    Slots beyond the number of eligible papers have index ``-1``.
    ``exclude[i]`` is a document query ``i`` must never retrieve (``-1`` = none), e.g. the
    paper whose title was used as the query.
    With ``out_dir`` the results are written to ``.npy`` memmaps there.
    """
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    n, m = len(query_emb), len(doc_emb)

    order = np.argsort(doc_year, kind="stable")          # documents sorted by year
    sorted_year = doc_year[order]
    sorted_pos = np.empty(m, np.int64)
    sorted_pos[order] = np.arange(m)
    D = torch.from_numpy(np.ascontiguousarray(doc_emb[order])).to(device=device, dtype=dtype)
    order_t = torch.from_numpy(order).to(device)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cand_idx = np.lib.format.open_memmap(out_dir / "cand_idx.npy", "w+", np.int32, (n, top_m))
        cand_sim = np.lib.format.open_memmap(out_dir / "cand_sim.npy", "w+", np.float16, (n, top_m))
    else:
        cand_idx = np.empty((n, top_m), np.int32)
        cand_sim = np.empty((n, top_m), np.float16)

    q_order = np.argsort(query_year, kind="stable")       # batches share similar prefixes
    with torch.no_grad():
        for b in tqdm(range(0, n, batch_size), desc="retrieve", unit="batch"):
            rows = q_order[b:b + batch_size]
            limit = np.searchsorted(sorted_year, query_year[rows], side="right")   # eligible prefix per query
            L = int(limit.max())
            if L == 0:
                cand_idx[rows] = -1
                cand_sim[rows] = 0
                continue
            Q = torch.from_numpy(np.ascontiguousarray(query_emb[rows])).to(device=device, dtype=dtype)
            S = Q @ D[:L].T
            cols = torch.arange(L, device=device)
            S.masked_fill_(cols[None, :] >= torch.from_numpy(limit).to(device)[:, None], float("-inf"))
            if exclude is not None:
                ex = exclude[rows]
                hit = np.nonzero(ex >= 0)[0]
                pos_ex = sorted_pos[ex[hit]]
                keep = pos_ex < L
                hit, pos_ex = hit[keep], pos_ex[keep]
                S[torch.from_numpy(hit).to(device), torch.from_numpy(pos_ex).to(device)] = float("-inf")
            k = min(top_m, L)
            vals, pos = torch.topk(S, k, dim=1)          # sorted descending
            idx = order_t[pos].to(torch.int32)
            idx[torch.isinf(vals)] = -1
            vals = vals.float().masked_fill(torch.isinf(vals), 0.0)
            out_i = np.full((len(rows), top_m), -1, np.int32)
            out_s = np.zeros((len(rows), top_m), np.float16)
            out_i[:, :k] = idx.cpu().numpy()
            out_s[:, :k] = vals.cpu().numpy().astype(np.float16)
            cand_idx[rows] = out_i
            cand_sim[rows] = out_s
    if out_dir is not None:
        cand_idx.flush()
        cand_sim.flush()
    return cand_idx, cand_sim
