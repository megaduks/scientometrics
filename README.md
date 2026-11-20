master_thesis
=============

Does the empirical citation distribution simply follow from how researchers search for papers?

Hypothesis: an author searches a scholarly search engine with keywords related to their paper. They cite
a few papers from the first result page, fewer from the second page, and so on. `citesim` simulates
this behaviour on a real corpus and produces a simulated citation distribution to compare with the
empirical one.

Data
----

`notebooks/scientometrics_quotes.ipynb` filters AMiner DBLP v10 (`data/external/dblp-ref-10/`, 3.08M papers)
to venues on the Polish ministry journal/conference list and attaches their `gov_score`. This gives
850,406 papers in `data/interim/articles_with_score_df.csv`. `data/` is tracked with DVC (`dvc pull`).

The **empirical target** is `internal_citations`: how many times each paper is cited by other papers
*of the same corpus*, counted from their reference lists. It lives in the same closed world as the
simulation and has no artefacts. AMiner's own `n_citation` counts citations from outside the corpus
and uses 50 as a placeholder for "unknown" (30% of papers). It is exported too, but don't use it as
the target.

Generator (`citesim/`)
----------------------

    python -m citesim.run prepare                                   # embed + search, cached (GPU recommended)
    python -m citesim.run simulate --grid configs/grid_default.json --workers 8
    python -m citesim.run collect                                   # -> simulated_distributions.csv
    pytest tests

1. **Twins.** Each real paper gets one hypothetical twin with the same publication year and the same
   number of in-corpus references.
2. **Queries.** Each twin gets a title-like keyword query (`--query-source generated`, the default),
   e.g. "Optimizing Distributed Query Scheduler". The template is
   `{verb-ing} {adjective} {noun} {agent}`, filled uniformly from the lists in `data/raw/`. With
   `--query-source titles`, the real paper's own title is the query (and the paper itself is excluded).
3. **Search.** Titles and queries are embedded with `all-MiniLM-L6-v2` (unit vectors, cosine similarity).
   For every query, the top `--top-m` (default 500) most similar papers *published no later than the
   twin* are retrieved by exact search.
4. **Ranking.** The candidates are re-ranked by
   `w_sim·similarity + w_rec·recency + w_cit·log(1+citations) + w_ven·venue_points`.
   Each feature is min-max normalised within the candidate set, so the weights are comparable.
5. **Citing.** Twins are processed in chronological order. The results are cut into pages of
   `page_size`, and the twin cites distinct papers without replacement:
   - `per_page = [3, 1, 1]` means 3 papers chosen at random from page 1, 1 from page 2, 1 from page 3;
   - `per_page = null` means geometric mode: `k` papers, page p weighted `exp(-decay·p)`, where
     `k = null` uses the twin's real reference count.
6. **Citation signal.** With `citation_signal = "dynamic"` (default), the citation feature is the count
   accumulated so far in the simulation: what the search engine would show at that time.
   `"static"` ranks by the final empirical counts instead. It is circular and kept only for comparison.

Each run writes `params.json`, `counts.npy` (citations per paper, aligned with the CSV rows) and
`distribution.json` (`{citations: papers}`, zeros included) under
`data/processed/citesim/<retrieval>/runs/<run_id>/`. Runs are seeded and skipped if already present.
