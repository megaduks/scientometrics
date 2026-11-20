"""Citation-behaviour simulation.

Hypothesis under test: the empirical citation distribution comes from how researchers
search. They query a search engine with keywords related to their own paper, then cite
papers from the first result page, fewer from the second page, and so on.

Pipeline:
1. ``data``     load the corpus and compute within-corpus citation counts from reference lists.
2. ``embed``    embed paper titles and queries (sentence-transformers, unit-normalised).
3. ``queries``  generate keyword queries, one hypothetical "twin" per real paper.
4. ``retrieve`` top-M most similar earlier-or-same-year papers for every query.
5. ``simulate`` re-rank the candidates, paginate, cite, and accumulate citation counts.
"""
