import csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy.special import expit
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = str(ROOT / "submission/predict27/generated") + "/"
INK, MUTED, GRID = "#222222", "#666666", "#dddddd"
C = dict(blue="#0072B2", green="#009E73", orange="#D55E00", purple="#CC79A7", yellow="#E69F00", navy="#332288")

# Fig: unknown-scale example
fig, ax = plt.subplots(figsize=(4.6, 1.9))
paths = {"(1,1)": ([0, 1, 2], C["blue"], 25/26), "(2,0)": ([0, 2, 2], C["orange"], 9/10)}
for name, (x, col, post) in paths.items():
    ax.plot([0, 1, 2], expit(np.array(x)), color=col, lw=2, marker="o", ms=5, label=f"increments {name}: history posterior {post:.3f}")
ax.axhline(expit(2), color=GRID, lw=1, zorder=0)
ax.text(2.06, expit(2)-.06, f"common endpoint\nendpoint posterior {5**1.5/(1+5**1.5):.3f}", va="center", fontsize=7, color=MUTED)
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["$t=0$", "$t=1$", "$t=2$"]); ax.set_ylabel("price"); ax.set_xlim(-.1, 3.1)
ax.legend(frameon=False, loc="lower right", fontsize=7)
fig.tight_layout(); fig.savefig(OUT + "scale_example.pdf"); plt.close(fig)

# Fig: paired differences, full scale and zoomed
rows = [r for r in csv.DictReader(open(ROOT / "code/feedback_results/contrasts.csv")) if r["epsilon"] == "0.001" and r["metric"] == "brier"]
get = {(int(r["lead"]), r["contrast"]): (float(r["mean"]), float(r["lo"]), float(r["hi"])) for r in rows}
left = [("mixture-recent", "Mixture"), ("mixture_no_volume-recent", "Mixture, no volume"), ("gaussian_history-recent", "Gaussian history"),
        ("history_logistic-recent", "History logistic"), ("latest-recent", "Latest price"), ("calibrated-recent", "Linear price calibration")]
right = [("combined-recent", "Price + mixture"), ("combined_no_volume-recent", "Price + mixture, no volume"),
         ("gaussian_combined-recent", "Price + Gaussian"), ("history_logistic-endpoint_matched", "History logistic vs\nmatched endpoint$^\\dagger$")]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1, 1]})
for ax, items, title in ((axes[0], left, "Standalone forecasts"), (axes[1], right, "Incremental comparisons (zoomed)")):
    for k, (key, lab) in enumerate(items):
        for lead, col, dy in ((5, C["blue"], -.13), (1, C["orange"], .13)):
            m, lo, hi = get[(lead, key)]
            ax.plot([lo, hi], [k+dy]*2, color=col, lw=2, solid_capstyle="round")
            ax.plot(m, k+dy, "o", color=col, ms=5, mec="white", mew=.8, label=f"{lead}-minute lead" if k == 0 else None)
    ax.axvline(0, color=MUTED, lw=.8)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([l for _, l in items]); ax.invert_yaxis()
    ax.set_xlabel("Brier difference vs reference"); ax.set_title(title, fontsize=8); ax.grid(axis="x", color=GRID, lw=.5)
axes[0].legend(frameon=False, fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(OUT + "empirical_differences_v2.pdf"); plt.close(fig)

# Fig: date-level differences, price + mixture minus cubic control
rep = json.load(open(ROOT / "code/feedback_results/report.json"))
fig, ax = plt.subplots(figsize=(4.6, 1.9))
for lead, col, dx in ((5, C["blue"], -.15), (1, C["orange"], .15)):
    d = [r for r in rep["per_date"] if r["lead"] == lead and r["contrast"] == "combined-recent"]
    d.sort(key=lambda r: r["date"])
    ax.plot(np.arange(len(d))+dx, [r["diff"] for r in d], "o", color=col, ms=4.5, mec="white", mew=.6, label=f"{lead}-minute lead")
    labels = [r["date"][5:] for r in d]
ax.axhline(0, color=MUTED, lw=.8)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=60, fontsize=6.5)
ax.set_ylabel("Brier difference"); ax.grid(axis="y", color=GRID, lw=.5); ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower center", bbox_to_anchor=(.5, 1.0))
fig.tight_layout(); fig.savefig(OUT + "date_differences.pdf"); plt.close(fig)

# Fig: oracle endpoint and contamination
o = json.load(open(ROOT / "code/feedback_results/oracle.json"))
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
a = o["part_a"]; T = [r["T"] for r in a]
axes[0].plot(T, [r["intrinsic_brier"] for r in a], "-o", color=C["blue"], lw=2, ms=5, label="history over oracle endpoint")
axes[0].plot(T, [r["calib_brier"] for r in a], "-o", color=C["orange"], lw=2, ms=5, label="fitted calibrator error")
axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[0].set_xticks(T); axes[0].set_xticklabels(T); axes[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axes[0].set_xlabel("observed increments $T$"); axes[0].set_ylabel("Brier risk component"); axes[0].legend(frameon=False, fontsize=7)
axes[0].set_title("Synthetic mixture, fixed design", fontsize=8)
st = {r["eps"]: r for r in json.load(open(ROOT / "code/feedback_results/contamination_stratified.json"))}
pb = {r["eps"]: r for r in o["part_b"] if r["eps"] > 0}
eps = sorted(pb)
g = [st[e]["gain_log"] if e in st else pb[e]["gain_log"] for e in eps]
se = [st[e]["se_log"] if e in st else pb[e]["se_log"] for e in eps]
axes[1].errorbar(eps, g, yerr=[2*x for x in se], fmt="-o", color=C["green"], lw=2, ms=5, capsize=2, label="$I(Y;Z\\mid S)$, nats")
axes[1].set_xscale("log"); axes[1].set_yscale("log"); axes[1].set_xlabel("Student-$t_5$ weight $\\varepsilon$")
axes[1].set_ylabel("gain from history (nats)")
axes[1].set_title("Aligned Gaussian with $t_5$ contamination, $T=10$", fontsize=8)
for ax in axes: ax.grid(color=GRID, lw=.5)
fig.tight_layout(); fig.savefig(OUT + "oracle_endpoint.pdf"); plt.close(fig)
print("ok")
