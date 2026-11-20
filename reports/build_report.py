"""Assemble reports/report.html from figures/*.png and report_data.json."""
import base64
import json
from pathlib import Path

R = Path(__file__).resolve().parent
D = json.loads((R / "report_data.json").read_text())


def img(name, alt):
    b64 = base64.b64encode((R / "figures" / name).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}">'


def fig(name, num, caption, alt):
    return (f'<figure class="plate"><div class="plate-img">{img(name, alt)}</div>'
            f'<figcaption><span class="fignum">Figure {num}.</span> {caption}</figcaption></figure>')


def table(head, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tbl {cls}"><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


pct = lambda x: f"{100 * x:.1f}%"
w = lambda r: f'<span class="mono">({r["w_sim"]:g}, {r["w_rec"]:g}, {r["w_cit"]:g}, {r["w_ven"]:g})</span>'
d, rc, pool, fe = D["data"], D["recall"], D["pool"], D["formal_emp"]

pilot_rows = [["Empirical (target)", f'{d["internal_citations"]:,}', pct(d["never_cited"]), f'{d["p99"]:.0f}',
               f'{d["p999"]:.0f}', f'{d["max_internal"]:,}']]
for k, v in D["pilot"].items():
    pilot_rows.append([k.capitalize(), f'{v["total"]:,}', pct(v["never"]), f'{v["p99"]:.0f}', f'{v["p999"]:.0f}',
                       f'{v["max"]:,}'])

s1_rows = [[r["source"], r["citation_signal"], w(r), pct(r["zeros"]), f'{r["p99"]:.0f}', f'{r["p999"]:.0f}',
            f'{r["max"]:,}', f'{r["ks"]:.3f}'] for r in D["stage1_best"]]

s2_rows = [[f'{int(r["top_m"]):,}', r["citation_signal"], int(r["page_size"]), f'<span class="mono">{r["per_page"]}</span>',
            w(r), pct(r["zeros"]), f'{r["p99"]:.0f}', f'{r["p999"]:.0f}', f'{r["max"]:,}', f'<b>{r["ks"]:.3f}</b>']
           for r in D["stage2_top"]]

seed_rows = [[f'{int(r["top_m"]):,}', r["citation_signal"], int(r["page_size"]),
              f'<span class="mono">{r["per_page"]}</span>', w(r), int(r["n"]), f'{r["zeros_sd"]:.4f}',
              f'{r["p99_sd"]:.2f}', f'{r["max_sd"]:.1f}', f'{r["ks_sd"]:.4f}'] for r in D["seeds"]]

fm_rows = []
for r in D["formal"]:
    wt = r["w"].replace("[", "(").replace("]", ")")
    fm_rows.append([f'{int(r["top_m"]):,}', r["signal"], int(r["page"]), f'<span class="mono">{r["per_page"]}</span>',
                    f'<span class="mono">{wt}</span>', pct(r["never_cited"]), f'{r["max"]:,.0f}',
                    f'{r["alpha_hat"]:.2f} <span class="dim">(+{r["delta_alpha"]:.2f})</span>',
                    r["AD_reject"], r["LR_reject"], f'{r["vuong_R"]:.1f}', r["twosample_reject"], f'{r["ks_shape"]:.3f}'])

fm_rows.sort(key=lambda x: float(x[7].split()[0]))
fmed = D["stage2_factor_medians"]

html = f"""<title>Citesim Experiment Report</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Serif:wght@500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column (~70ch) with figures and tables allowed to run wider; plates stay white in both themes. */
:root {{
  --paper: #f6f7f9; --ink: #16181d; --ink-2: #474b55; --ink-3: #6b6f7a; --rule: #d9dce2;
  --accent: #2a78d6; --accent-ink: #1d5fae; --warm: #c4521f; --panel: #eceff4; --plate: #ffffff;
  --display: "IBM Plex Serif", Georgia, "Times New Roman", serif;
  --body: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --paper: #121419; --ink: #e7e9ee; --ink-2: #b8bcc6; --ink-3: #8d929d; --rule: #2c3038;
  --accent: #5c9ce8; --accent-ink: #8ab8f0; --warm: #ee8a5c; --panel: #1b1e25; --plate: #ffffff; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --paper: #121419; --ink: #e7e9ee; --ink-2: #b8bcc6; --ink-3: #8d929d; --rule: #2c3038;
  --accent: #5c9ce8; --accent-ink: #8ab8f0; --warm: #ee8a5c; --panel: #1b1e25; --plate: #ffffff; color-scheme: dark; }}
body {{ background: var(--paper); color: var(--ink); font: 16px/1.65 var(--body); }}
.wrap {{ max-width: 1120px; margin: 0 auto; padding-block: 48px 96px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }}
.col {{ max-width: 72ch; }}
header {{ border-bottom: 1px solid var(--rule); padding-bottom: 28px; margin-bottom: 36px; }}
.kicker {{ font: 500 12px/1 var(--mono); letter-spacing: .12em; text-transform: uppercase; color: var(--accent-ink); }}
h1 {{ font: 600 clamp(30px, 4.4vw, 46px)/1.12 var(--display); margin: 14px 0 12px; text-wrap: balance; max-width: 24ch; }}
.meta {{ color: var(--ink-3); font-size: 14px; }}
h2 {{ font: 600 26px/1.25 var(--display); margin: 64px 0 14px; text-wrap: balance; }}
h2 .sec {{ font: 500 14px var(--mono); color: var(--accent-ink); margin-right: 10px; vertical-align: 3px; }}
h3 {{ font: 600 18px/1.35 var(--body); margin: 36px 0 8px; }}
p, li {{ color: var(--ink); }}
p {{ margin: 0 0 14px; }}
ul, ol {{ padding-left: 22px; margin: 0 0 16px; }} li {{ margin-bottom: 6px; }}
a {{ color: var(--accent-ink); }}
.mono, code {{ font-family: var(--mono); font-size: .88em; }}
code {{ background: var(--panel); padding: 1px 5px; border-radius: 3px; }}
.dim {{ color: var(--ink-3); }}
.summary {{ background: var(--panel); border-radius: 6px; padding: 22px 26px; margin: 8px 0 12px; }}
.summary h2 {{ margin-top: 0; font-size: 20px; }}
.summary li {{ margin-bottom: 8px; }}
.verdict {{ display: inline-block; font: 500 12px/1 var(--mono); letter-spacing: .06em; text-transform: uppercase;
  padding: 5px 8px; border-radius: 3px; border: 1px solid var(--warm); color: var(--warm); margin-right: 6px; }}
.verdict.ok {{ border-color: var(--accent); color: var(--accent-ink); }}
nav.toc {{ columns: 2 260px; column-gap: 32px; font-size: 14.5px; margin: 24px 0 0; }}
nav.toc a {{ color: var(--ink-2); text-decoration: none; display: block; padding: 3px 0; break-inside: avoid; }}
nav.toc a:hover, nav.toc a:focus-visible {{ color: var(--accent-ink); text-decoration: underline; }}
.plate {{ margin: 28px 0 32px; }}
.plate-img {{ background: var(--plate); border: 1px solid var(--rule); border-radius: 6px; padding: 12px; }}
.plate img {{ display: block; width: 100%; height: auto; max-width: 100%; }}
figcaption {{ font-size: 14px; color: var(--ink-2); margin-top: 10px; max-width: 90ch; line-height: 1.55; }}
.fignum {{ font-weight: 600; color: var(--ink); }}
.tbl {{ overflow-x: auto; margin: 18px 0 26px; border: 1px solid var(--rule); border-radius: 6px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13.5px; font-variant-numeric: tabular-nums; }}
th {{ text-align: left; font: 500 11.5px/1.3 var(--mono); letter-spacing: .04em; text-transform: uppercase; color: var(--ink-3);
  background: var(--panel); padding: 9px 11px; border-bottom: 1px solid var(--rule); white-space: nowrap; }}
td {{ padding: 8px 11px; border-bottom: 1px solid var(--rule); vertical-align: top; }}
tr:last-child td {{ border-bottom: 0; }}
.tbl.wide td:nth-child(n+2) {{ white-space: nowrap; }}
.eq {{ font-family: var(--mono); font-size: 14px; background: var(--panel); padding: 12px 16px; border-radius: 6px; overflow-x: auto; margin: 0 0 16px; white-space: pre; }}
.grid2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px 32px; }}
.grid2 > * {{ min-width: 0; }}
.callout {{ border-left: 3px solid var(--accent); padding: 4px 0 4px 16px; margin: 18px 0; color: var(--ink-2); }}
footer {{ margin-top: 72px; border-top: 1px solid var(--rule); padding-top: 18px; color: var(--ink-3); font-size: 13.5px; }}
@media (max-width: 560px) {{ body {{ font-size: 15.5px; }} .summary {{ padding: 18px; }} }}
</style>

<div class="wrap">
<header>
  <div class="kicker">Scientometrics · citation simulation · report 1</div>
  <h1>Can search behaviour explain the citation distribution?</h1>
  <div class="meta">Experiments run on 20 November 2026 · corpus: AMiner DBLP v10 (850,406 papers) · code: <span class="mono">github.com/megaduks/scientometrics</span></div>
  <nav class="toc" aria-label="Contents">
    <a href="#summary">Summary</a><a href="#hypothesis">1 · Hypothesis</a><a href="#data">2 · Input data</a>
    <a href="#audit">3 · The inherited code</a><a href="#model">4 · Simulation model</a><a href="#plan">5 · Experimental plan</a>
    <a href="#pilot">6.1 · Pilot runs</a><a href="#stage1">6.2 · Stage 1</a><a href="#stage2">6.3 · Stage 2</a>
    <a href="#pool">6.4 · Candidate pool</a><a href="#formal">6.5 · Formal tests</a><a href="#interpretation">7 · Interpretation</a>
    <a href="#limits">8 · Limitations</a><a href="#next">9 · Further research</a><a href="#appendix">Appendices</a>
  </nav>
</header>

<section class="summary col" id="summary">
<h2>Summary</h2>
<ul>
<li><span class="verdict ok">body reproduced</span>A simulated researcher who searches by topic, ranks results partly by citation count, and cites from the first one or two result pages reproduces the empirical citation distribution up to about the 99.9th percentile. The best settings leave about 35% of papers never cited (empirical: 36.3%), and the largest gap between the cumulative distributions is 0.014.</li>
<li><span class="verdict">tail not reproduced</span>All four formal tests from the collaborators' plan reject every shortlisted setting (8 settings × 3 seeds). Above x<sub>min</sub> = 88 the simulated tails are steeper (α̂ = 2.90–4.80 against α₀ = 2.78) and curved like a log-normal, while the empirical tail is a straight power law. The most-cited simulated paper reaches 3,034 citations at most; the real one has 4,420.</li>
<li>The key ingredients are realistic topics (queries taken from real titles, not from a fixed word list) and a moderate citation weight (0.30–0.35 when counts accumulate during the simulation, 0.40 with real counts). The citing rule, page size, recency and venue weights matter much less.</li>
<li>Enlarging the search result pool from 500 to 2,000 candidates roughly doubles the most-cited count and moves the tail exponent towards the empirical value. This suggests the remaining gap comes from limited exposure: a paper can only be cited by queries that retrieve it.</li>
<li>The original student code could not be used: the generator that produced the earlier simulated data is lost, and the surviving version ranked the <i>least</i> similar papers first. Everything here comes from a rewritten, tested generator (<span class="mono">citesim</span>).</li>
</ul>
</section>

<section class="col" id="hypothesis">
<h2><span class="sec">1</span>Hypothesis and research question</h2>
<p>The hypothesis is that the heavy-tailed distribution of citations is, to a large extent, a product of how researchers find what they cite. A researcher writing a paper searches a scholarly search engine (such as Google Scholar) with keywords related to the paper. They then cite a few papers from the first result page, fewer from the second page, and so on. The engine's ranking mixes topical relevance with signals such as recency, citation count and venue prestige.</p>
<p>Real Google Scholar cannot be queried at scale, so the process is simulated on a real database of computer-science papers. Many hypothetical new papers each run a search, re-rank the results by a weighted score, paginate them, and cite papers from the top pages. The citations they generate form a simulated citation distribution.</p>
<p class="callout"><b>Research question.</b> Is the simulated distribution of citations the same as the empirical one? Only the <i>shape</i> of the distribution matters, not which particular papers get cited.</p>
<p>The statistical comparison follows the plan in the e-mail from the Institute of Mathematics (PUT). Treat the empirical tail as a fully specified Pareto/power law. Test the simulated tails against it with Anderson-Darling (sensitive in the tails). Then test the exponent with a likelihood-ratio test, and compare the power law with a log-normal using Vuong's test.</p>
</section>

<section id="data">
<div class="col">
<h2><span class="sec">2</span>Input data</h2>
<p><b>Corpus.</b> The AMiner citation network, DBLP version 10, holds 3,079,007 computer-science papers with titles, years, venues and reference lists. Papers are kept only if their venue exactly matches an entry on the Polish ministry list of journals and conferences, which also provides a venue score (20–200 points). This leaves <b>{d["papers"]:,} papers</b> published between {d["year_min"]} and {d["year_max"]}. The subset is intentional: it is the database the simulated researchers search.</p>
<p><b>Choosing the empirical target.</b> AMiner provides a citation count per paper (<code>n_citation</code>), but it has two problems. First, it counts citations from the whole literature, not from the corpus being searched. Second, it uses 50 as a placeholder for unknown counts: {d["at50"]:,} papers ({pct(d["share50"])}) have exactly 50 (Figure 1A). Papers with the placeholder are cited only {d["internal_of_50"]:.1f} times on average from inside the corpus. The target is therefore the number of times each paper is cited <b>by other papers in the corpus</b>, counted from their reference lists. It lives in the same closed world as the simulation and has no artefacts. On a log scale it correlates {d["corr"]:.2f} with <code>n_citation</code> (placeholder excluded).</p>
</div>
{table(["Quantity", "Value"], [
    ["Papers in corpus", f'{d["papers"]:,}'],
    ["Within-corpus citations (target)", f'{d["internal_citations"]:,}'],
    ["Papers never cited", pct(d["never_cited"])],
    ["99th / 99.9th percentile (cited papers)", f'{d["p99"]:.0f} / {d["p999"]:.0f}'],
    ["Most-cited paper", f'{d["max_internal"]:,} (Lowe, SIFT, 2004)'],
    ["References to corpus papers, mean / median", f'{d["mean_refs"]:.2f} / {d["median_refs"]:.0f}'],
    ["Papers citing nothing in the corpus", pct(d["zero_refs"])],
    ["AMiner n_citation = 50 (placeholder)", f'{d["at50"]:,} ({pct(d["share50"])})'],
])}
{fig("fig1_data.png", 1, "A: AMiner's <code>n_citation</code> around 50 (log scale). The bar at exactly 50 is the placeholder for unknown counts. B: complementary cumulative distributions (CCDF, P(X ≥ x)) of the two citation measures, over cited papers. The placeholder produces the vertical drop in the blue curve at 50. The black curve is the target used throughout.", "Bar chart of n_citation values 40 to 60 with a spike at 50, and log-log CCDFs of the two citation measures")}
</section>

<section id="audit">
<div class="col">
<h2><span class="sec">3</span>Starting point: the inherited code</h2>
<p>The project started from a master's thesis repository (J. Fiturski) with four notebooks and two files of results: <code>source_distribution.csv</code> (empirical) and <code>distribution_citations.csv</code> (306 simulated distributions). The audit found that the results cannot be trusted or reproduced.</p>
</div>
{table(["Finding", "Consequence"], [
    ["The code that produced <code>distribution_citations.csv</code> is not in the repository. Its 306 runs (850,000 queries × k ∈ {10, 25, 50}) carry no record of their parameters.", "The earlier simulated data cannot be reproduced or interpreted. The student is no longer reachable."],
    ["The surviving generator used ChromaDB <i>distances</i> as “similarity” with a positive weight.", "Ranking favoured the least similar papers: the cited papers sat at ranks ~175,000–199,900 of 200,000."],
    ["The generated queries were never executed; only <code>\"Example query\"</code> was run.", "No search behaviour was actually simulated in the saved notebook."],
    ["The random seed was reset on every call, and citations were drawn with replacement.", "Every hypothetical paper made identical page choices and could cite the same paper twice."],
    ["Citations were ranked by final AMiner counts, with no time order.", "This mixes the target into the ranking (circular)."],
    ["<code>powerlaw.ipynb</code> fitted and tested the <i>frequency lists</i> (papers per count), not the citation counts, and hard-coded one value.", "Its conclusions (“all distributions are power laws”, “shape almost identical”) are unsupported."],
], "wide")}
<p class="col">The repository was cleaned of unused template code and moved to <span class="mono">megaduks/scientometrics</span>. The generator was then rewritten from scratch as a tested Python package (Section 4). Appendix A records what the legacy files show, for completeness.</p>
</section>

<section id="model">
<div class="col">
<h2><span class="sec">4</span>The simulation model (<span class="mono">citesim</span>)</h2>
<ol>
<li><b>Hypothetical papers.</b> A run writes N new papers (default N = 850,406, one per corpus paper). Each borrows a real "source" paper, which supplies its title for title queries and its reference count where needed.</li>
<li><b>Query.</b> Either a <i>generated</i> keyword query, of the form “{{verb-ing}} {{adjective}} {{noun}} {{agent}}” (e.g. “Optimizing Distributed Query Scheduler”), built from four word lists of 100, 100, 182 and 100 words; or the source paper's real <i>title</i>, with the source paper excluded from its own results.</li>
<li><b>Search.</b> Titles and queries are embedded with <span class="mono">all-MiniLM-L6-v2</span> (unit vectors). Exact cosine search over the whole corpus on the GPU keeps the top M candidates (M = 500 or 2,000).</li>
<li><b>Ranking.</b> The candidates are re-ranked by a weighted score. Each feature is min-max scaled within the candidate set, so the weights are comparable:</li>
</ol>
<div class="eq">score = w_sim·similarity + w_rec·recency(year) + w_cit·log(1 + citations) + w_ven·venue points</div>
<ol start="5">
<li><b>Citation signal.</b> <i>Static</i> ranks by the papers' real within-corpus counts, as a search engine shows them today. <i>Dynamic</i> ranks by citations received so far from the run's own hypothetical papers, starting from zero (a rich-get-richer feedback). Hypothetical papers are processed in random order.</li>
<li><b>Pagination and citing.</b> The ranked list is cut into pages of 10 or 20. Each paper cites distinct papers chosen at random within each page, following a citing rule: [5] means five from page 1; [3,1,1] means three, one and one from pages 1–3; [2,1,1,1] and [1,1,1,1,1] spread citations further. Every rule cites 5 papers, close to the real mean of {d["mean_refs"]:.2f} in-corpus references.</li>
<li><b>Output.</b> Citation counts for all 850,406 papers, including zeros. The results are compared with the empirical target.</li>
</ol>
<p>Each run is seeded and stores its parameters. The package has 17 unit tests covering, for example, that similarity ranks the most similar paper first, that no paper is cited twice by the same hypothetical paper, that runs are reproducible, and that the goodness-of-fit test holds its level. An optional <code>--dated</code> mode restricts each search to papers no newer than the source paper; it was used only in the pilot runs.</p>
</div>
{table(["Step", "Hardware / cost"], [
    ["Embed 850k titles (once, cached)", "GTX 1650 Ti 4 GB · about 9 min"],
    ["Embed 850k generated queries", "about 4.5 min"],
    ["Exact top-M search, whole corpus, 850k queries", "about 9 min (M = 500) · 9.5 min (M = 2,000) · 2.5 GB / 10 GB on disk"],
    ["One simulation run (850k papers)", "about 2 min alone · about 4 min each with 8 in parallel"],
])}
</section>

<section id="plan">
<div class="col">
<h2><span class="sec">5</span>Experimental plan</h2>
<p>The experiment ran in stages, each one narrowing the parameter space for the next. The shape distance used for screening is the largest absolute gap between the simulated and empirical CCDFs, evaluated on a log-spaced grid from 1 to 10,000 citations (0 = identical).</p>
</div>
{table(["Stage", "Question", "Design", "Runs", "Wall time"], [
    ["Pilot", "Does the pipeline work? How do query sources differ?", "1 setting × 2 query sources, time-restricted search", "2", "about 5 min"],
    ["Stage 1 (screening)", "Which query source and citation weight?", "20 weight vectors (w_cit 0–0.75, w_rec and w_ven 0 or 0.1) × static/dynamic × 2 query sources; page 10, [3,1,1]", "80", "56 min"],
    ["Stage 2 (refinement)", "Fine weights, citing rule, page size, noise", "Title queries, M = 500: 16 dynamic (w_cit 0.2–0.35) + 12 static (w_cit 0.4–0.6) weight vectors × page 10/20 × 4 citing rules; seeds 0–2 for the 4 best stage-1 settings", "224", "1 h 41 min"],
    ["Stage 2 pool", "Does a larger candidate pool lengthen the tail?", "The same 28 weight vectors at M = 2,000; page 10, [3,1,1]", "28", "26 min"],
    ["Replicates", "Seed variability of the shortlist", "Seeds 1–2 for the 8 best settings", "16", "11 min"],
    ["Formal tests", "Are the distributions the same?", "A-D, LR, Vuong and two-sample A-D on 8 settings × 3 seeds", "24 tested", "a few min"],
])}
</section>

<section id="results">
<div class="col"><h2><span class="sec">6</span>Results</h2></div>

<div id="pilot">
<div class="col">
<h3>6.1 Pilot: generated queries versus title queries</h3>
<p>Both pilot runs used the same settings: weights (0.5, 0.3, 0.1, 0.1), page 10, [3,1,1], dynamic citation counts, with search restricted to earlier papers. They produce almost the same number of citations but opposite shapes (Figure 2). With generated queries the citations pile onto few papers: 79% are never cited, and the five most-cited papers are all about debugging, because the 482-word vocabulary is dominated by software-engineering terms. With real titles as queries the citations spread far too evenly: the most-cited paper gets 270 citations.</p>
</div>
{table(["Run", "Citations", "Never cited", "99th pct", "99.9th pct", "Most-cited"], pilot_rows)}
{fig("fig3_pilot.png", 2, "CCDFs of the two pilot runs against the empirical distribution. Generated queries concentrate citations too much; title queries spread them too evenly.", "Log-log CCDF of empirical, generated-query and title-query pilot runs")}
<div class="col">
<p><b>How well does topical search recover real citations?</b> For 20,000 sampled papers the paper's own title was used as the query and the 500 results were compared with its real reference list. Only <b>{pct(rc["share"])}</b> of the real in-corpus references appear in the top 500. Of those, {pct(rc["top10"])} are on page 1 (top 10) and {pct(rc["top30"])} in the top 30; the median rank is {rc["median_rank"]:.0f}. With the time-restricted search the share was 44%. Topical similarity of titles explains part of real citing, but most references are not among the most similar titles.</p>
</div>
{fig("fig2_references.png", 3, f"A: number of references each paper makes to other corpus papers (mean {d['mean_refs']:.2f}, median {d['median_refs']:.0f}; {pct(d['zero_refs'])} cite nothing in the corpus). B: rank of real references inside the paper's own title search, for the {pct(rc['share'])} that appear in the top 500.", "Histogram of in-corpus reference counts and histogram of ranks of real references")}
</div>

<div id="stage1">
<div class="col">
<h3>6.2 Stage 1: query source and citation weight</h3>
<p>From here on, every hypothetical paper searches the whole database. The citation weight is the main dial (Figure 4). Without it, title queries spread citations too thin (about 10% never cited). At high weight, citations concentrate on a few papers (over 80% never cited). The empirical shape falls between, at about w_cit = 0.25 with dynamic counts or 0.5 with static counts. No weight rescues the generated queries: their best run still leaves 77% of papers uncited, and adding citation weight only makes it worse. The four settings without a citation term produced identical static and dynamic runs, which serves as a reproducibility check.</p>
</div>
{fig("fig4_stage1.png", 4, "Stage 1 (80 runs). Lines are the median over the four recency/venue combinations; bands show their range. A: shape distance to the empirical distribution. B: share of papers never cited; the dotted line is the empirical 36%.", "Two line charts of shape distance and never-cited share against citation weight, by query source and citation signal")}
{table(["Query source", "Signal", "Weights (sim, rec, cit, ven)", "Never cited", "99th pct", "99.9th pct", "Most-cited", "Distance"], s1_rows, "wide")}
</div>

<div id="stage2">
<div class="col">
<h3>6.3 Stage 2: refining weights, citing rule and page size</h3>
<p>Stage 2 used title queries only. It narrows the citation weight to a clear optimum: 0.30–0.35 with dynamic counts and 0.40 with static counts (Figure 5). Within that optimum, flatter citing rules ([5] and [1,1,1,1,1]) beat [3,1,1], and 20 results per page is slightly better than 10 (Figure 6). Across the stage-2 design, the median distance by citing rule is {", ".join(f'{k} {v:.3f}' for k, v in fmed["per_page"].items())}; by signal it is dynamic {fmed["citation_signal"]["dynamic"]:.3f} and static {fmed["citation_signal"]["static"]:.3f}. Recency and venue weights change the median distance by less than 0.01. The best settings reach a shape distance of 0.014. One setting gets the 99.9th percentile almost exactly (262 vs 263) but still stops at 997 citations for its most-cited paper.</p>
</div>
{fig("fig5_stage2.png", 5, "Stage 2 (title queries, 500 candidates, seed 0). Shape distance against citation weight for each citing rule (colour) and page size (marker and line style); bands span the recency/venue variants. A: dynamic signal. B: static signal.", "Line charts of shape distance against citation weight for four citing rules and two page sizes")}
{fig("fig6_rules.png", 6, "Shape distance for runs inside the optimum (dynamic w_cit 0.30–0.35, static w_cit 0.40), by citing rule (A) and page size (B). Each dot is one run.", "Box plots of shape distance by citing rule and by page size")}
{table(["Pool", "Signal", "Page", "Rule", "Weights", "Never cited", "99th", "99.9th", "Most-cited", "Distance"], s2_rows, "wide")}
<div class="col">
<p><b>Seed variability is negligible.</b> For the settings run with three seeds, the shape distance varies by at most 0.001 and the never-cited share by under 0.1 percentage points. Differences between settings are therefore real, not noise.</p>
</div>
{table(["Pool", "Signal", "Page", "Rule", "Weights", "Seeds", "SD never cited", "SD 99th", "SD most-cited", "SD distance"], seed_rows, "wide")}
</div>

<div id="pool">
<div class="col">
<h3>6.4 Candidate pool: 500 versus 2,000 results</h3>
<p>All stage-1 and stage-2 runs have one thing in common: their most-cited paper falls far short of the empirical 4,420. One explanation is exposure. A paper can only be cited by queries that put it among their M candidates. Repeating the 28 weight vectors with M = 2,000 tests this (Figure 7). The most-cited count rises by a median factor of {pool["max_ratio_median"]:.2f}, and the highest value across runs grows from {pool["max_overall_500"]:,} to {pool["max_overall_2000"]:,}. For dynamic settings near the optimum, the shape distance also improves (best 0.018–0.020 at [3,1,1] against 0.021–0.047 at M = 500). Static settings get worse, because the larger pool also concentrates citations.</p>
</div>
{fig("fig7_pool.png", 7, "Paired comparison of the same settings (page 10, [3,1,1], seed 0) with 500 and 2,000 candidates; the dashed line marks no change. A: most-cited paper (log scales), with the empirical maximum dotted. B: shape distance.", "Scatter plots comparing most-cited count and shape distance between pool sizes")}
</div>

<div id="formal">
<div class="col">
<h3>6.5 Formal tests on the shortlist</h3>
<p><b>Shortlist.</b> The 8 settings with the smallest shape distance (6 with M = 500, 2 with M = 2,000), each run with 3 seeds, give 24 runs in total.</p>
<p><b>Null model.</b> The tail of the empirical target is fitted once with a discrete power law, P(x) = x<sup>−α</sup> / ζ(α, x<sub>min</sub>) for x ≥ x<sub>min</sub>, with x<sub>min</sub> chosen by the Clauset-Shalizi-Newman KS method. The fitted values are then treated as true: <b>x<sub>min</sub> = {fe["xmin"]}, α₀ = {fe["alpha0"]:.3f} ± {fe["se0"]:.3f}</b>, with {fe["n_tail"]:,} papers in the tail. The empirical tail fits its own law very well (A² = {fe["A2_self"]:.2f}, p = {fe["p_self"]:.2f}; optimistic, because the parameters come from the same data). Vuong's test cannot separate it from a log-normal (R = {fe["vuong_R"]:.2f}, p = {fe["vuong_p"]:.2f}).</p>
<p><b>Tests per run</b> (Benjamini-Hochberg across the 24 runs, separately per test):</p>
<ol>
<li><b>Anderson-Darling against the fully specified law.</b> Citation counts are integers with many ties, so a continuous A-D gives infinite statistics. The discrete A² of Choulakian, Lockhart and Stephens (1994) is used instead, with a 999-sample Monte Carlo null. Under H0 it rejected {pct(fe["cal5"])} at the 5% level (100 replicates), so it is slightly liberal.</li>
<li><b>Likelihood-ratio test</b> of α = α₀ against a freely estimated α, compared with χ²(1).</li>
<li><b>Vuong's test</b> of power law against log-normal on the tail above x<sub>min</sub> (non-nested).</li>
<li><b>Two-sample Anderson-Darling</b> between the simulated and empirical tails, with no model assumed (999 permutations).</li>
</ol>
<p><b>Result: every run is rejected by every test.</b> The A² values range from 11 to 520, far above anything the null produces. Figure 8 shows why: the simulated curves follow the empirical one closely up to roughly 100–200 citations, then bend down and stop between about 380 and 1,300 citations, while the empirical tail continues straight to 4,420. Every simulated exponent is above α₀, and Vuong favours the log-normal in all 24 runs (Figure 9). The two settings with the 2,000 pool come closest (α̂ = 2.92 and 2.99; A² 12–16). The static settings, which match the body best, have the worst tails.</p>
</div>
{fig("fig8_shortlist.png", 8, "CCDFs of the eight shortlisted settings (seed 0) against the empirical distribution. The two settings closest in the tail, both with 2,000 candidates, are highlighted. The dotted line marks x<sub>min</sub> = 88.", "Log-log CCDFs of eight simulated settings and the empirical distribution")}
{fig("fig9_formal.png", 9, "Formal test statistics for the 8 settings × 3 seeds. A: tail exponent α̂ against the empirical α₀ (shaded: ±1.96 SE). B: Vuong normalised log-likelihood ratio (negative favours the log-normal); the empirical value is −0.83. C: discrete Anderson-Darling distance from the fitted empirical law (log scale). Orange: the two settings with 2,000 candidates.", "Three dot plots of tail exponent, Vuong R and Anderson-Darling statistic per setting")}
{table(["Pool", "Signal", "Page", "Rule", "Weights", "Never cited", "Most-cited", "α̂ (Δα)", "A-D rejects", "LR rejects", "Vuong R", "2-sample rejects", "Distance"], fm_rows, "wide")}
</div>
</section>

<section class="col" id="interpretation">
<h2><span class="sec">7</span>Interpretation</h2>
<p><b>What the search mechanism explains.</b> Three ingredients are enough to reproduce the bulk of the empirical distribution, about 99.5% of all papers. Researchers search by the topic of their own paper. The engine boosts already-cited papers moderately (the citation term carries about a third of the weight). Researchers cite fairly evenly from the first one or two pages. The never-cited share, the median and the 99th percentile all match. The citation feedback is essential: without it, citations spread too evenly; with too much of it, a few papers absorb everything. The topical structure is also essential: a narrow, artificial query vocabulary cannot be compensated by any ranking.</p>
<p><b>What it does not explain.</b> The extreme tail, the few hundred papers with more than ~300 citations, does not emerge. The simulated tails bend in a log-normal way. This is the pattern expected when growth is multiplicative but each item's exposure is capped. Here a paper's audience is limited to the queries for which it is topically among the top M results, and the citation boost reorders papers only within that pool. Enlarging the pool moved the tail in the right direction, which supports this reading. The empirical classics (SIFT, LIBSVM, Random Forests, the SVM paper) are general-purpose methods cited from almost every subfield; their titles are not similar to the titles of the papers that cite them.</p>
<p><b>Answer to the research question.</b> On the evidence so far, the simulated and empirical distributions are <i>not</i> the same. The formal tests reject equality for every setting tried. Search-and-paginate behaviour with citation-boosted ranking is a sufficient explanation for the body of the distribution, but not for its power-law tail. An additional mechanism that gives some papers audience beyond their topic seems necessary.</p>
</section>

<section class="col" id="limits">
<h2><span class="sec">8</span>Limitations and threats to validity</h2>
<ul>
<li><b>Closed world.</b> Both the target and the simulation count only citations within the 850k-paper subset defined by the ministry venue list. Results may differ for the full literature.</li>
<li><b>Titles as topics.</b> Queries and documents are represented by titles only, embedded with a small sentence model. Abstracts would give a richer topical signal; only 38.6% of real references appear in a title search's top 500.</li>
<li><b>Static ranking reads the target.</b> Static runs rank by the real within-corpus counts, so the target enters the simulation. Dynamic runs avoid this, and they match better.</li>
<li><b>Fixed citing volume.</b> Every hypothetical paper cites exactly 5 papers. Real reference lists vary (mean 4.93, 22% cite nothing in the corpus).</li>
<li><b>Exposure cap.</b> Only the top M = 500 or 2,000 results per query are ever ranked. This is a computational shortcut that, as shown above, shapes the tail.</li>
<li><b>Test sensitivity.</b> With 3,000–5,300 papers in each tail, the tests detect small differences. The effect sizes (Δα, the CCDF gap) should be read together with the p-values. The Monte Carlo A-D test is slightly liberal (8% at the 5% level), which does not affect any decision.</li>
<li><b>One random query set.</b> Generated queries used one seed; title queries are deterministic. Query-sampling variability was not measured.</li>
</ul>
</section>

<section class="col" id="next">
<h2><span class="sec">9</span>Further research</h2>
<ol>
<li><b>Test the exposure explanation directly.</b> Run the best dynamic settings with M = 5,000 and with an uncapped citation channel. One version: rank globally by citations a fraction of the time. Another: re-rank the full result list rather than the top M. If the tail straightens, the hypothesis needs only a broader ranking, not a new mechanism. An M = 5,000 cache needs about 25 GB; 37 GB is free.</li>
<li><b>Add a “classics” channel.</b> Let a small share of citations go to papers found through a topic-independent route (most-cited lists, surveys, textbooks, citing what others cite). Estimate the share that straightens the tail, and test whether it is plausible.</li>
<li><b>Model citation copying.</b> Cite some papers from the reference lists of papers already chosen (the Simkin-Roychowdhury mechanism). Copying is a known source of power-law tails and is close to how researchers actually work.</li>
<li><b>Richer topical representation.</b> Use abstracts for embeddings and multi-query searches per paper. Measure how much of the real reference list each variant recovers, and calibrate against it.</li>
<li><b>Realistic citing volume.</b> Draw each hypothetical paper's number of citations from the empirical reference-count distribution, and let the citing rule scale with it.</li>
<li><b>Time dynamics.</b> Run the dated mode with dynamic counts in true chronological order, and compare the simulated and real citation trajectories of individual cohorts, not only the final distribution.</li>
<li><b>Formal inference over parameters.</b> Replace grid screening with approximate Bayesian computation or a surrogate model. Use the CCDF gap and the tail statistics as summaries, to obtain a posterior over weights and rules and a principled model comparison.</li>
<li><b>Generated queries.</b> If the original keyword design is to stay in the study, build the word lists from the corpus titles themselves (thousands of words, weighted by frequency), so that queries cover the database the way real topics do.</li>
</ol>
</section>

<section id="appendix">
<div class="col">
<h2><span class="sec">A</span>Appendix A: analyses of the legacy data</h2>
<p>Before the generator was rewritten, the tests from the e-mail were also run on the inherited files: the empirical distribution was AMiner <code>n_citation</code> with the placeholder dropped, and the simulated data were the 306 legacy distributions. They were run as test cases for a reusable testing procedure. The results are recorded here, but the legacy simulations come from unknown, buggy code, so they carry no evidential weight.</p>
</div>
{table(["Test (legacy data)", "Result"], [
    ["Empirical tail (n_citation, 50s dropped)", "x_min = 587, α = 2.58 ± 0.02, 5,314 papers; power law vs log-normal not separable (Vuong p ≈ 0.24–0.27)"],
    ["Anderson-Darling vs fully specified Pareto", "306 / 306 runs rejected"],
    ["LR test of α = α₀", "283–290 / 306 runs rejected; 16–23 runs (mostly rows 10–32) consistent"],
    ["Vuong, power law vs log-normal (6 sampled runs)", "log-normal favoured in 5–6 / 6"],
], "wide")}
<div class="col">
<p>These runs also showed two technical points for any analysis of citation counts. First, continuous tests fail on integer data with ties at x<sub>min</sub>, so discrete versions are needed. Second, the <span class="mono">powerlaw</span> package caps α at 3 by default, so steep tails must be fitted with a wider range. Both are handled in <span class="mono">citesim.gof</span>.</p>

<h2><span class="sec">B</span>Appendix B: reproducibility</h2>
<p>Code: <span class="mono">github.com/megaduks/scientometrics</span> (commits <span class="mono">b9c4bfc</span> clean-up, <span class="mono">2aaec28</span> generator, <span class="mono">d504e92</span> whole-database search, <span class="mono">19c2d33</span> stage-2 grids). Results are cached under <code>data/processed/citesim/</code>; the formal tests are in <code>notebooks/formal_tests.ipynb</code>.</p>
<div class="eq">P="python -m citesim.run --query-source titles"
$P prepare                                   # embed + exact search (cached)
$P simulate --grid configs/grid_stage2.json --workers 8
$P --top-m 2000 prepare
$P --top-m 2000 simulate --grid configs/grid_stage2_pool2000.json --workers 8
python reports/make_figures.py && python reports/build_report.py</div>
</div>
</section>

<footer>Prepared with Claude Code for the scientometrics project. Every number in this report is computed from the stored run outputs by <span class="mono">reports/make_figures.py</span>.</footer>
</div>
"""
(R / "report.html").write_text(html)
print("report.html", round(len(html) / 1e6, 2), "MB")
