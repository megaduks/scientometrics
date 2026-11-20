import numpy as np
import pandas as pd
import pytest

from citesim.data import _count_internal, distribution
from citesim.queries import gerund
from citesim.retrieve import retrieve
from citesim.run import expand_grid
from citesim.simulate import SimConfig, simulate


def unit(rows):
    x = np.asarray(rows, np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# docs 0..4 on a quarter circle; years increase with index
DOCS = unit([[1, 0], [0.9, 0.1], [0.7, 0.7], [0.1, 0.9], [0, 1]])
YEARS = np.array([2000, 2001, 2002, 2003, 2004], np.int16)


def test_retrieve_most_similar_first_and_no_future_papers():
    q = unit([[0, 1], [1, 0]])
    idx, sim = retrieve(q, np.array([2004, 2004], np.int16), DOCS, YEARS, top_m=5, device="cpu")
    assert idx[0, 0] == 4 and idx[1, 0] == 0          # nearest first (similarity, not distance)
    assert np.all(np.diff(sim[0].astype(float)) <= 1e-3)

    idx, _ = retrieve(q, np.array([2001, 2001], np.int16), DOCS, YEARS, top_m=5, device="cpu")
    assert set(idx[0][idx[0] >= 0]) == {0, 1}          # only papers from <= 2001
    assert list(idx[0][2:]) == [-1, -1, -1]            # padding when too few are eligible


def test_retrieve_excludes_source_paper():
    idx, _ = retrieve(DOCS, YEARS, DOCS, YEARS, top_m=5, exclude=np.arange(5), device="cpu")
    for i in range(5):
        assert i not in idx[i]
    assert np.all(idx[0] == -1)                        # the oldest paper has nothing earlier to cite


def test_retrieve_whole_database_with_sampled_sources():
    src = np.array([4, 4, 1])                          # titles of docs 4, 4, 1 used as queries
    every = np.full(3, YEARS.max())                    # no time constraint
    idx, _ = retrieve(DOCS[src], every, DOCS, YEARS, top_m=5, exclude=src, device="cpu")
    for row, s in zip(idx, src):
        assert s not in row and np.all(row[:4] >= 0) and row[4] == -1
    assert idx[0, 0] == 3 and idx[2, 0] == 0             # nearest remaining neighbours first


def _toy(n_queries=50, m=30, top_m=20, seed=1):
    rng = np.random.default_rng(seed)
    cand = np.stack([rng.permutation(m)[:top_m] for _ in range(n_queries)]).astype(np.int32)
    sim = np.sort(rng.random((n_queries, top_m)), axis=1)[:, ::-1].astype(np.float16)
    return dict(cand_idx=cand, cand_sim=sim, query_year=np.full(n_queries, 2010),
                query_k=rng.integers(0, 8, n_queries), doc_year=rng.integers(1990, 2010, m),
                doc_venue=rng.choice([20.0, 100.0, 200.0], m),
                doc_static_citations=rng.integers(0, 100, m))


def test_similarity_only_cites_top_result():
    t = _toy()
    cfg = SimConfig(weights=(1, 0, 0, 0), page_size=1, per_page=(1,))
    counts = simulate(cfg, **t)
    expected = np.bincount(t["cand_idx"][:, 0], minlength=len(counts))
    assert np.array_equal(counts, expected)


def test_per_page_counts_and_no_duplicates():
    t = _toy(n_queries=1)
    counts = simulate(SimConfig(page_size=5, per_page=(5, 3, 2)), **t)
    assert counts.sum() == 10 and counts.max() == 1


def test_geometric_uses_twin_reference_counts():
    t = _toy()
    counts = simulate(SimConfig(per_page=None, k=None, decay=0.7), **t)
    assert counts.sum() == np.minimum(t["query_k"], 20).sum()


def test_seeded_and_reproducible():
    t = _toy()
    a = simulate(SimConfig(seed=3), **t)
    assert np.array_equal(a, simulate(SimConfig(seed=3), **t))
    assert not np.array_equal(a, simulate(SimConfig(seed=4), **t))


def test_dynamic_citations_feed_back():
    t = _toy(n_queries=40)
    t["cand_idx"][:] = np.arange(20)                  # everyone sees the same candidates
    t["cand_sim"][:] = 0.5
    counts = simulate(SimConfig(weights=(0, 0, 1, 0), page_size=1, per_page=(1,)), **t)
    assert counts[0] == 40                             # first cited paper keeps winning


def test_count_internal_ignores_self_and_outside_refs():
    ids = pd.Series(["a", "b", "c"])
    refs = pd.Series(["['b', 'c', 'zz']", "['b', 'c', 'c']", "[]"])
    cited, refs_in = _count_internal(ids, refs)
    assert list(cited) == [0, 1, 2] and list(refs_in) == [2, 1, 0]
    assert distribution(cited) == {0: 1, 1: 1, 2: 1}


@pytest.mark.parametrize("verb,expected", [("Analyze", "Analyzing"), ("Map", "Mapping"), ("Cache", "Caching"),
                                           ("Query", "Querying"), ("Debug", "Debugging"), ("Build", "Building")])
def test_gerund(verb, expected):
    assert gerund(verb) == expected


def test_expand_grid_blocks_and_dedup():
    grid = [{"page_size": [10, 20], "per_page": [[3, 1]]}, {"page_size": [10], "per_page": [[3, 1]]}]
    assert len(expand_grid(grid)) == 2
