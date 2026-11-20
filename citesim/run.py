"""Command-line entry point.

    python -m citesim.run prepare  [--query-source generated|titles] [--top-m 500]
    python -m citesim.run simulate --grid configs/grid_default.json [--workers 8]
    python -m citesim.run collect  [--out data/processed/citesim/simulated_distributions.csv]

``prepare`` is the expensive step (embedding plus exact search). It is cached and shared by
all simulation runs. Every run writes ``params.json``, ``counts.npy`` (citations per paper)
and ``distribution.json`` (``{citations: papers}``, zeros included).
"""
from __future__ import annotations

import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .data import distribution, load_corpus
from .simulate import SimConfig, simulate

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data/interim/articles_with_score_df.csv"
DEFAULT_RAW = ROOT / "data/raw"
DEFAULT_CACHE = ROOT / "data/processed/citesim"


def _retrieval_dir(cache: Path, query_source: str, top_m: int, seed: int) -> Path:
    return cache / (f"{query_source}_m{top_m}" + ("" if query_source == "titles" else f"_s{seed}"))


def cmd_prepare(a):
    from .embed import cached_embeddings
    from .queries import generate_queries, load_word_lists
    from .retrieve import retrieve

    cache = Path(a.cache)
    corpus = load_corpus(a.csv, cache)
    print(f"corpus: {len(corpus)} papers, {corpus.internal_citations.sum()} internal citations")
    for name, counts in [("internal", corpus.internal_citations), ("n_citation", corpus.n_citation)]:
        (cache / f"empirical_{name}.json").write_text(json.dumps(distribution(counts)))

    title_emb = cached_embeddings(corpus.titles.tolist(), cache / "title_emb.npy", device=a.device)
    out = _retrieval_dir(cache, a.query_source, a.top_m, a.seed)
    out.mkdir(parents=True, exist_ok=True)
    if a.query_source == "generated":
        queries = generate_queries(len(corpus), load_word_lists(a.raw), np.random.default_rng(a.seed))
        (out / "queries.txt").write_text("\n".join(queries) + "\n")
        query_emb = cached_embeddings(queries, out / "query_emb.npy", device=a.device)
    else:
        query_emb = title_emb
    # Query i is the twin of paper i: same publication year, same number of in-corpus references.
    np.save(out / "query_year.npy", corpus.year)
    np.save(out / "query_k.npy", corpus.internal_refs)
    retrieve(query_emb, corpus.year, title_emb, corpus.year, a.top_m,
             exclude_self=a.query_source == "titles", device=a.device, out_dir=out)
    print("prepared", out)


def expand_grid(spec: dict | list) -> list[SimConfig]:
    """A grid is a dict of lists (cartesian product) or a list of such dicts."""
    blocks = spec if isinstance(spec, list) else [spec]
    configs = []
    for block in blocks:
        keys = list(block)
        for values in itertools.product(*(block[k] for k in keys)):
            configs.append(SimConfig(**dict(zip(keys, values))))
    return list({c.run_id: c for c in configs}.values())


def _run_one(cfg: SimConfig, retrieval: str, cache: str, runs: str) -> str:
    rdir, cache_p = Path(retrieval), Path(cache)
    z = np.load(cache_p / "corpus.npz", allow_pickle=True)
    counts = simulate(
        cfg,
        np.load(rdir / "cand_idx.npy", mmap_mode="r"),
        np.load(rdir / "cand_sim.npy", mmap_mode="r"),
        np.load(rdir / "query_year.npy"),
        np.load(rdir / "query_k.npy"),
        z["year"], z["gov_score"], z["internal_citations"],
    )
    out = Path(runs) / cfg.run_id
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "counts.npy", counts.astype(np.int32))
    (out / "distribution.json").write_text(json.dumps(distribution(counts)))
    (out / "params.json").write_text(json.dumps({**asdict(cfg), "retrieval": rdir.name,
                                                 "total_citations": int(counts.sum())}))
    return cfg.run_id


def cmd_simulate(a):
    configs = expand_grid(json.loads(Path(a.grid).read_text()))
    rdir = _retrieval_dir(Path(a.cache), a.query_source, a.top_m, a.seed)
    runs = Path(a.runs or rdir / "runs")
    todo = [c for c in configs if not (runs / c.run_id / "params.json").exists()]
    print(f"{len(configs)} configs, {len(todo)} to run -> {runs}")
    with ProcessPoolExecutor(a.workers) as pool:
        futures = [pool.submit(_run_one, c, str(rdir), a.cache, str(runs)) for c in todo]
        for i, f in enumerate(as_completed(futures), 1):
            print(f"[{i}/{len(todo)}] {f.result()}", flush=True)


def cmd_collect(a):
    import pandas as pd

    rdir = _retrieval_dir(Path(a.cache), a.query_source, a.top_m, a.seed)
    runs = Path(a.runs or rdir / "runs")
    rows = []
    for p in sorted(runs.glob("*/params.json")):
        params = json.loads(p.read_text())
        params["distribution"] = (p.parent / "distribution.json").read_text()
        rows.append(params)
    out = Path(a.out or rdir / "simulated_distributions.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{len(rows)} runs -> {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="citesim")
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--query-source", choices=["generated", "titles"], default="generated")
    ap.add_argument("--top-m", type=int, default=500, help="candidates kept per query")
    ap.add_argument("--seed", type=int, default=0, help="query-generation seed")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--csv", default=str(DEFAULT_CSV))
    p.add_argument("--raw", default=str(DEFAULT_RAW))
    p.add_argument("--device", default=None)
    p.set_defaults(func=cmd_prepare)
    s = sub.add_parser("simulate")
    s.add_argument("--grid", required=True)
    s.add_argument("--runs", default=None)
    s.add_argument("--workers", type=int, default=8)
    s.set_defaults(func=cmd_simulate)
    c = sub.add_parser("collect")
    c.add_argument("--runs", default=None)
    c.add_argument("--out", default=None)
    c.set_defaults(func=cmd_collect)
    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
