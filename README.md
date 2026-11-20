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

Global options (before the subcommand) select one cached retrieval: `--query-source generated|titles`,
`--n-papers N` (default: one per corpus paper), `--dated`, `--top-m M` (default 500) and `--seed S`.
Use the same options for `prepare`, `simulate` and `collect`.

    python -m citesim.run [options] prepare                          # embed + search, cached (GPU recommended)
    python -m citesim.run [options] simulate --grid configs/grid_stage1.json --workers 8
    python -m citesim.run [options] collect                          # -> simulated_distributions.csv
    pytest tests

1. **Hypothetical papers.** A run writes `--n-papers` new papers, one search query each. Each borrows a
   random real "source" paper (each real paper exactly once when N equals the corpus size). It takes the
   source's number of in-corpus references (used by geometric mode with `k = null`) and, with
   `--query-source titles`, its title.
2. **Queries.** By default (`--query-source generated`) each paper gets a title-like keyword query,
   e.g. "Optimizing Distributed Query Scheduler". The template is `{verb-ing} {adjective} {noun} {agent}`,
   filled uniformly from the lists in `data/raw/`. With `--query-source titles`, the source paper's
   title is the query, and the source paper itself is never retrieved.
3. **Search.** Titles and queries are embedded with `all-MiniLM-L6-v2` (unit vectors, cosine similarity).
   Exact search returns the `--top-m` most similar papers from the **whole database**: every
   hypothetical paper is new and may cite any existing paper. With `--dated`, the search is limited to
   papers no newer than the source paper's year.
4. **Ranking.** The candidates are re-ranked by
   `w_sim·similarity + w_rec·recency + w_cit·log(1+citations) + w_ven·venue_points`.
   Each feature is min-max normalised within the candidate set, so the weights are comparable.
5. **Citing.** The results are cut into pages of `page_size`, and the paper cites distinct papers
   without replacement:
   - `per_page = [3, 1, 1]` means 3 papers chosen at random from page 1, 1 from page 2, 1 from page 3;
   - `per_page = null` means geometric mode: `k` papers, page p weighted `exp(-decay·p)`.
6. **Citation signal.** `"static"` ranks by the papers' real (within-corpus) citation counts, as a
   search engine shows them today. `"dynamic"` ranks by citations received so far from this run's
   hypothetical papers, starting at zero. Papers are processed in random order (by year, then random,
   with `--dated`).

Grids: `configs/grid_stage1.json` is the screening grid. It has 20 weight vectors (citation weight
0 to 0.75, recency and venue weights 0 or 0.1, similarity takes the rest) × static/dynamic × page
size 10 × `[3, 1, 1]`, which is 40 runs per retrieval. Run it for both query sources.
Stage 1 showed that title queries match best, with citation weight about 0.25 (dynamic) or 0.5 (static).
`configs/grid_stage2.json` (title queries, `--top-m 500`, 232 runs) refines around that point:
citation weight 0.2–0.35 with dynamic counts and 0.4–0.6 with static counts, × recency/venue 0 or 0.1
× page size 10/20 × citing rules [5], [3,1,1], [2,1,1,1], [1,1,1,1,1]. It adds seeds 0–2 for the four
best stage-1 settings. `configs/grid_stage2_pool2000.json` (28 runs) repeats the weight grid at
[3,1,1]/10 with `--top-m 2000`, to test whether a larger result pool lengthens the extreme tail.
`configs/grid_default.json` is the older, broader grid.

Each run writes `params.json`, `counts.npy` (citations per paper, aligned with the CSV rows) and
`distribution.json` (`{citations: papers}`, zeros included) under
`data/processed/citesim/<retrieval>/runs/<run_id>/`. Runs are seeded and skipped if already present.
