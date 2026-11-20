"""Figures and tables for the experiment report (reports/report.html).

Run from experiment/master_thesis:  ../../.venv/bin/python reports/make_figures.py
Writes reports/figures/*.png and reports/report_data.json.
"""
import ast
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/processed/citesim"
FIG = ROOT / "reports/figures"
FIG.mkdir(parents=True, exist_ok=True)

# Palette: validated categorical slots (blue, orange, aqua, violet) + ink for the empirical series.
INK, BLUE, ORANGE, AQUA, VIOLET, GREY = "#16181d", "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#b7b9bf"
sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95,
              rc={"axes.edgecolor": "#c9cbd1", "grid.color": "#e6e7ea", "axes.titleweight": "semibold",
                  "axes.titlesize": 12, "axes.labelcolor": "#3b3e46", "xtick.color": "#4a4d55",
                  "ytick.color": "#4a4d55", "legend.frameon": False, "figure.dpi": 100,
                  "axes.spines.top": False, "axes.spines.right": False})
D = {}  # numbers for the report text


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=170, facecolor="white")
    plt.close(fig)


def ccdf(x):
    v, c = np.unique(x, return_counts=True)
    tail = np.cumsum(c[::-1])[::-1] / len(x)
    return v, tail


def load_counts(retr, run_id):
    return np.load(BASE / retr / "runs" / run_id / "counts.npy").astype(np.int64)


# ---------------------------------------------------------------- data
z = np.load(BASE / "corpus.npz")
emp, ncit, refs, year = (z["internal_citations"].astype(np.int64), z["n_citation"].astype(np.int64),
                         z["internal_refs"].astype(np.int64), z["year"])
D["data"] = dict(papers=int(len(emp)), internal_citations=int(emp.sum()), never_cited=float(np.mean(emp == 0)),
                 max_internal=int(emp.max()), p99=float(np.percentile(emp[emp > 0], 99)),
                 p999=float(np.percentile(emp[emp > 0], 99.9)), mean_refs=float(refs.mean()),
                 median_refs=float(np.median(refs)), zero_refs=float(np.mean(refs == 0)),
                 at50=int((ncit == 50).sum()), share50=float(np.mean(ncit == 50)),
                 internal_of_50=float(emp[ncit == 50].mean()), year_min=int(year.min()), year_max=int(year.max()),
                 corr=float(np.corrcoef(np.log1p(emp[ncit != 50]), np.log1p(ncit[ncit != 50]))[0, 1]))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
xs = np.arange(40, 61)
cnt = np.array([(ncit == v).sum() for v in xs])
ax[0].bar(xs, cnt, color=[ORANGE if v == 50 else BLUE for v in xs], width=0.8)
ax[0].set_yscale("log")
ax[0].set(xlabel="AMiner n_citation", ylabel="papers (log scale)", title="A. AMiner placeholder at 50 citations")
ax[0].annotate(f"{cnt[xs == 50][0]:,} papers\n(29.6% of corpus)", xy=(50, cnt[xs == 50][0]), xytext=(53, 6e4),
               fontsize=9, color="#3b3e46", arrowprops=dict(arrowstyle="-", color="#8a8d95"))
for x, lab, col, lw in [(emp, "within-corpus citations (target)", INK, 2.2), (ncit, "AMiner n_citation", BLUE, 1.4)]:
    v, t = ccdf(x[x > 0])
    ax[1].loglog(v, t, color=col, lw=lw, label=lab)
ax[1].set(xlabel="citations", ylabel="P(X ≥ x), cited papers", title="B. Two citation measures")
ax[1].legend(loc="lower left", fontsize=9)
save(fig, "fig1_data.png")

# ---------------------------------------------------------------- reference recall (titles, whole DB)
import pandas as pd  # noqa
cand = np.load(BASE / "titles_m500_n850406_all_s0/cand_idx.npy", mmap_mode="r")
raw = pd.read_csv(ROOT / "data/interim/articles_with_score_df.csv", usecols=["references"]).references
pos = {p: k for k, p in enumerate(z["ids"])}
rng = np.random.default_rng(0)
ranks, tot = [], 0
for i in rng.choice(len(emp), 20000, replace=False):
    r = raw[i]
    if not isinstance(r, str) or r == "[]":
        continue
    real = {pos[x] for x in ast.literal_eval(r) if x in pos and pos[x] != i}
    if not real:
        continue
    where = {int(v): k for k, v in enumerate(np.asarray(cand[i])) if v >= 0}
    tot += len(real)
    ranks += [where[j] for j in real if j in where]
ranks = np.array(ranks)
D["recall"] = dict(refs=tot, found=int(len(ranks)), share=len(ranks) / tot, top30=float(np.mean(ranks < 30)),
                   top10=float(np.mean(ranks < 10)), median_rank=float(np.median(ranks)))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
sns.histplot(np.clip(refs, 0, 30), discrete=True, color=BLUE, ax=ax[0], edgecolor="white", linewidth=0.6)
ax[0].set(xlabel="references to other corpus papers (30 = 30+)", ylabel="papers",
          title="A. Reference lists inside the corpus")
sns.histplot(ranks, bins=np.arange(0, 501, 10), color=AQUA, ax=ax[1], edgecolor="white", linewidth=0.4)
ax[1].set(xlabel="rank of a real reference in the paper's own title search (10 results per bin)",
          ylabel="references", title=f"B. {D['recall']['share']:.0%} of real references appear in the top 500")
save(fig, "fig2_references.png")

# ---------------------------------------------------------------- pilot runs (time-restricted)
pilot = {"generated queries": load_counts("generated_m500_n850406_dated_s0", "edcfdb093ce3"),
         "title queries": load_counts("titles_m500_n850406_dated_s0", "edcfdb093ce3")}
fig, ax = plt.subplots(figsize=(7.5, 4.6))
v, t = ccdf(emp)
ax.loglog(v[v > 0], t[v > 0], color=INK, lw=2.4, label="empirical")
for (lab, x), col in zip(pilot.items(), [BLUE, ORANGE]):
    v, t = ccdf(x)
    ax.loglog(v[v > 0], t[v > 0], color=col, lw=1.6, label=lab)
ax.set(xlabel="citations", ylabel="P(X ≥ x)", title="Pilot: same settings, two query sources")
ax.legend(fontsize=9)
save(fig, "fig3_pilot.png")
D["pilot"] = {k: dict(total=int(x.sum()), never=float(np.mean(x == 0)), max=int(x.max()),
                      p99=float(np.percentile(x[x > 0], 99)), p999=float(np.percentile(x[x > 0], 99.9)))
              for k, x in pilot.items()}

# ---------------------------------------------------------------- stage 1
s1 = pd.read_csv(BASE / "stage1_summary.csv")
emp_row = s1[s1.retrieval == "EMPIRICAL"].iloc[0]
s1 = s1[s1.retrieval != "EMPIRICAL"].copy()
s1["query source"] = s1.source.map({"generated": "generated queries", "titles": "title queries"})
s1["citation signal"] = s1.citation_signal
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
kw = dict(x="w_cit", hue="query source", style="citation signal", palette=[BLUE, ORANGE], markers=True,
          dashes={"dynamic": "", "static": (4, 2)}, errorbar=("pi", 100), err_kws={"alpha": 0.12}, markersize=7)
sns.lineplot(data=s1, y="ks", ax=ax[0], legend=False, **kw)
ax[0].set(xlabel="citation weight w_cit", ylabel="shape distance (max CCDF gap)",
          title="A. Distance to the empirical distribution")
sns.lineplot(data=s1, y="zeros", ax=ax[1], **kw)
ax[1].axhline(emp_row.zeros, color=INK, lw=1.2, ls=":")
ax[1].text(0.76, emp_row.zeros - 0.045, "empirical 36%", fontsize=9, color=INK, ha="right")
ax[1].set(xlabel="citation weight w_cit", ylabel="share of papers never cited", title="B. Never-cited papers")
ax[1].legend(fontsize=8.5, loc="center left")
save(fig, "fig4_stage1.png")
D["stage1_best"] = s1.sort_values("ks").groupby("source").head(3)[
    ["source", "citation_signal", "w_sim", "w_rec", "w_cit", "w_ven", "zeros", "max", "p99", "p999", "ks"]
].to_dict("records")

# ---------------------------------------------------------------- stage 2
s2 = pd.read_csv(BASE / "stage2_summary.csv")
s2 = s2[s2.retrieval != "EMPIRICAL"].copy()
s2["top_m"] = s2.top_m.astype(int)
s2["page_size"] = s2.page_size.astype(int)
s2["seed"] = s2.seed.astype(int)
m5 = s2[(s2.top_m == 500) & (s2.seed == 0) & (((s2.citation_signal == "dynamic") & s2.w_cit.between(0.2, 0.35))
                                             | ((s2.citation_signal == "static") & s2.w_cit.between(0.4, 0.6)))].copy()
m5["citing rule"] = m5.per_page
m5["results per page"] = m5.page_size
order = ["[5]", "[3, 1, 1]", "[2, 1, 1, 1]", "[1, 1, 1, 1, 1]"]
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
for a, sig in zip(ax, ["dynamic", "static"]):
    d = m5[m5.citation_signal == sig]
    sns.lineplot(data=d, x="w_cit", y="ks", hue="citing rule", hue_order=order, style="results per page",
                 palette=[BLUE, ORANGE, AQUA, VIOLET], markers=True, errorbar=("pi", 100),
                 err_kws={"alpha": 0.08}, ax=a, legend=(sig == "static"), markersize=6)
    a.set(xlabel="citation weight w_cit", ylabel="shape distance", title=f"{'A' if sig == 'dynamic' else 'B'}. {sig} citation signal")
ax[1].legend(fontsize=8, title=None, loc="upper left", ncol=2)
ax[0].set_xticks([0.2, 0.25, 0.3, 0.35]); ax[1].set_xticks([0.4, 0.5, 0.6])
save(fig, "fig5_stage2.png")

sweet = s2[(s2.top_m == 500) & (s2.seed == 0) & (((s2.citation_signal == "dynamic") & s2.w_cit.isin([0.3, 0.35]))
                                                  | ((s2.citation_signal == "static") & (s2.w_cit == 0.4)))]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [2, 1]})
sns.boxplot(data=sweet, x="per_page", y="ks", order=order, color="#dbe7f7", linecolor=BLUE, ax=ax[0], width=0.5)
sns.stripplot(data=sweet, x="per_page", y="ks", order=order, color=BLUE, size=4, ax=ax[0], alpha=0.7)
ax[0].set(xlabel="citing rule (papers cited from page 1, 2, …)", ylabel="shape distance", title="A. Citing rule")
sns.boxplot(data=sweet, x="page_size", y="ks", color="#fbe1d6", linecolor=ORANGE, ax=ax[1], width=0.5)
sns.stripplot(data=sweet, x="page_size", y="ks", color=ORANGE, size=4, ax=ax[1], alpha=0.7)
ax[1].set(xlabel="results per page", ylabel="", title="B. Page size")
save(fig, "fig6_rules.png")
D["stage2_top"] = s2[s2.seed == 0].sort_values("ks").head(10)[
    ["top_m", "citation_signal", "page_size", "per_page", "w_sim", "w_rec", "w_cit", "w_ven",
     "zeros", "max", "p99", "p999", "ks"]].to_dict("records")
D["stage2_factor_medians"] = {f: s2[(s2.top_m == 500) & (s2.seed == 0) & s2.w_cit.between(0.2, 0.6)]
                              .groupby(f).ks.median().round(3).to_dict() for f in ["per_page", "page_size", "citation_signal"]}

# seeds
key = ["top_m", "citation_signal", "page_size", "per_page", "w_sim", "w_rec", "w_cit", "w_ven"]
rep = s2.groupby(key).filter(lambda g: g.seed.nunique() >= 3)
D["seeds"] = (rep.groupby(key).agg(n=("seed", "nunique"), ks_sd=("ks", "std"), max_sd=("max", "std"),
                                   p99_sd=("p99", "std"), zeros_sd=("zeros", "std"))
              .reset_index().to_dict("records"))

# ---------------------------------------------------------------- pool 500 vs 2000
pp = s2[(s2.page_size == 10) & (s2.per_page == "[3, 1, 1]") & (s2.seed == 0)]
k2 = ["citation_signal", "w_sim", "w_rec", "w_cit", "w_ven"]
pair = pp[pp.top_m == 500].set_index(k2)[["zeros", "max", "p999", "ks"]].join(
    pp[pp.top_m == 2000].set_index(k2)[["zeros", "max", "p999", "ks"]], how="inner", lsuffix="_500",
    rsuffix="_2000").reset_index()
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
for a, (c5, c2, lab, logs) in zip(ax, [("max_500", "max_2000", "most-cited paper", True),
                                       ("ks_500", "ks_2000", "shape distance", False)]):
    sns.scatterplot(data=pair, x=c5, y=c2, hue="citation_signal", palette={"dynamic": BLUE, "static": ORANGE},
                    s=55, ax=a, edgecolor="white", linewidth=0.8, legend=(not logs))
    lo, hi = min(pair[c5].min(), pair[c2].min()), max(pair[c5].max(), pair[c2].max())
    a.plot([lo, hi], [lo, hi], color="#8a8d95", lw=1, ls="--")
    if logs:
        a.set_xscale("log"); a.set_yscale("log")
        from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
        for axis in (a.xaxis, a.yaxis):
            axis.set_major_locator(FixedLocator([300, 500, 1000, 2000, 4000]))
            axis.set_minor_locator(NullLocator())
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        a.axhline(D["data"]["max_internal"], color=INK, lw=1, ls=":")
        a.text(lo * 1.05, D["data"]["max_internal"] * 0.82, "empirical most-cited: 4,420", fontsize=8.5, color=INK)
    a.set(xlabel=f"{lab}, 500 candidates", ylabel=f"{lab}, 2,000 candidates",
          title=("A. " if logs else "B. ") + lab.capitalize())
ax[1].legend(fontsize=9, title=None)
save(fig, "fig7_pool.png")
D["pool"] = dict(max_ratio_median=float((pair.max_2000 / pair.max_500).median()),
                 max_overall_500=int(pair.max_500.max()), max_overall_2000=int(pair.max_2000.max()),
                 ks_median_500=float(pair.ks_500.median()), ks_median_2000=float(pair.ks_2000.median()))

# ---------------------------------------------------------------- formal tests
ft = pd.read_csv(BASE / "formal_tests_runs.csv")
ft["setting"] = ft.apply(lambda r: f"M{r.top_m} · {r.signal[:3]} · p{r.page} · {r.per_page.replace(' ', '')}\n"
                                   f"w = {r.w.replace(' ', '')}", axis=1)
fs = ft.groupby("setting").agg(alpha=("alpha_hat", "mean"), R=("vuong_R", "mean"), ks=("ks_shape", "mean")).sort_values("alpha")
ALPHA0 = 2.7846; SE0 = 0.0289
fig, ax = plt.subplots(1, 3, figsize=(13, 5.2), sharey=True, gridspec_kw={"width_ratios": [1.3, 1, 1]})
order_s = list(fs.index)
best = set(order_s[:2])
pal = {s: (ORANGE if s in best else BLUE) for s in order_s}
sns.stripplot(data=ft, y="setting", x="alpha_hat", order=order_s, hue="setting", palette=pal, size=7, ax=ax[0],
              legend=False, jitter=0.15)
ax[0].axvspan(ALPHA0 - 1.96 * SE0, ALPHA0 + 1.96 * SE0, color=INK, alpha=0.08)
ax[0].axvline(ALPHA0, color=INK, lw=1.4)
ax[0].text(ALPHA0 + 0.06, len(order_s) - 0.55, "empirical α₀ = 2.78 (±1.96 SE)", fontsize=8.5, color=INK)
ax[0].set(xlabel="tail exponent α̂ (3 seeds)", ylabel="", title="A. Tail exponent")
sns.stripplot(data=ft, y="setting", x="vuong_R", order=order_s, hue="setting", palette=pal, size=7, ax=ax[1],
              legend=False, jitter=0.15)
ax[1].axvline(0, color="#8a8d95", lw=1, ls="--")
ax[1].axvline(-0.83, color=INK, lw=1.4)
ax[1].text(-1.1, len(order_s) - 0.55, "empirical R = −0.83", fontsize=8.5, color=INK, ha="right")
ax[1].set(xlabel="Vuong R (< 0 favours log-normal)", ylabel="", title="B. Power law vs log-normal")
sns.stripplot(data=ft, y="setting", x="A2", order=order_s, hue="setting", palette=pal, size=7, ax=ax[2],
              legend=False, jitter=0.15)
ax[2].set_xscale("log")
ax[2].set(xlabel="Anderson-Darling A² (log scale)", ylabel="", title="C. Distance from the fitted law")
ax[0].tick_params(axis="y", labelsize=8)
save(fig, "fig9_formal.png")

# CCDF of shortlisted settings (seed 0)
fig, ax = plt.subplots(figsize=(9, 5.4))
v, t = ccdf(emp)
ax.loglog(v[v > 0], t[v > 0], color=INK, lw=2.6, label="empirical", zorder=5)
sel = ft[ft.seed == 0]
from citesim.simulate import SimConfig  # noqa: E402
for r in sel.itertuples():
    cfg = SimConfig(weights=tuple(ast.literal_eval(r.w)), citation_signal=r.signal, page_size=int(r.page),
                    per_page=tuple(ast.literal_eval(r.per_page)), seed=0)
    x = load_counts(f"titles_m{r.top_m}_n850406_all_s0", cfg.run_id)
    v, t = ccdf(x)
    hl = r.setting in best
    ax.loglog(v[v > 0], t[v > 0], color=ORANGE if hl else GREY, lw=1.8 if hl else 1.0,
              label=r.setting.replace("\n", "  ") if hl else None, zorder=4 if hl else 2)
ax.plot([], [], color=GREY, lw=1, label="other shortlisted settings")
ax.axvline(88, color="#8a8d95", lw=1, ls=":")
ax.text(92, 2e-6, "x_min = 88", fontsize=8.5, color="#5b5e66")
ax.set(xlabel="citations", ylabel="P(X ≥ x)", title="Shortlisted settings vs the empirical distribution (seed 0)")
ax.legend(fontsize=8, loc="lower left")
save(fig, "fig8_shortlist.png")

D["formal"] = pd.read_csv(BASE / "formal_tests_summary.csv").to_dict("records")
D["formal_emp"] = dict(xmin=88, alpha0=ALPHA0, se0=SE0, n_tail=3814, A2_self=0.190, p_self=0.990,
                       vuong_R=-0.83, vuong_p=0.406, cal5=0.08, cal10=0.14)
(ROOT / "reports/report_data.json").write_text(json.dumps(D, indent=1, default=float))
print("figures:", sorted(p.name for p in FIG.glob("*.png")))
print(json.dumps({k: D[k] for k in ["data", "recall", "pilot", "pool"]}, indent=1, default=float))
