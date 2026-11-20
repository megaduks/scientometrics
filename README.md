master_thesis
=============

Simulating how researchers search for and cite papers, to check whether the resulting citation
distributions follow the same power law as real ones.

Pipeline (all in `notebooks/`):

1. `scientometrics_quotes.ipynb`: filter AMiner DBLP v10 papers (`data/external/dblp-ref-10/`) to venues on the
   Polish ministry journal/conference list and attach their `gov_score` (output: `data/interim/articles_with_score_df.csv`).
2. `scientometrics_search_engine.ipynb`: embed titles with `all-MiniLM-L6-v2` and load them into ChromaDB; extract
   word lists from titles with spaCy.
3. `scientometrics_queries_generator.ipynb`: generate synthetic queries from the word lists in `data/raw/`.
4. `distribution_generation_test.ipynb`: rank query results by weighted criteria (similarity, year, citations,
   venue score), sample citations with a page-position decay, and produce simulated citation distributions.

Setup
-----

    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    dvc pull                     # data/ is tracked with DVC (DagsHub remote)
    docker compose up -d         # ChromaDB on localhost:8000

Run notebooks from the `notebooks/` directory; they use relative `../data/...` paths.
