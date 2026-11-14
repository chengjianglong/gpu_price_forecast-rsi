"""Static figures for the README (docs/figures/*.png).

    python scripts/figures.py

Inputs: Pacific's pack (series built by gpr.pack), the starter demo's reports
(data/derived/demo, from run_demo.sh), and results/long_run.json. Colours are
the validated reference palette (blue #2a78d6, orange #eb6834 on #fcfcfb);
every figure with two series carries a legend and direct labels.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpr import pack as packlib  # noqa: E402

OUT = ROOT / "docs" / "figures"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2 = "#2a78d6", "#eb6834"          # categorical slots 1 and 2 (validated, light mode)
BAND1, BAND2 = "#cde2fb", "#fbd9c9"    # light tints of each series for ranges

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["system-ui", "-apple-system", "Helvetica Neue", "Arial", "sans-serif"],
    "font.size": 11, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left",
    "axes.titlesize": 13, "axes.titleweight": "semibold", "legend.frameon": False,
})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def fig_baselines() -> None:
    demo = ROOT / "data" / "derived" / "demo"
    methods = [("last_value_empirical", "last value + empirical range"), ("drift", "drift"),
               ("seasonal_naive", "seasonal naive")]
    cut = [json.loads((demo / f"report_frozen_{m}.json").read_text())["overall"]["skill_pooled"] for m, _ in methods]
    back = [json.loads((demo / "backtest" / f"report_{m}.json").read_text())["overall"]["skill_pooled"] for m, _ in methods]
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    y = np.arange(len(methods))
    h = 0.36
    for off, vals, color, label in ((-h / 2 - 0.01, cut, S1, "24 Oct cutoff"), (h / 2 + 0.01, back, S2, "15 weekly origins, 15 Jul–24 Oct")):
        ax.barh(y + off, vals, height=h, color=color, label=label, zorder=3)
        for yi, v in zip(y + off, vals):
            ax.text(v + (0.008 if v >= 0 else -0.008), yi, f"{v:+.2f}", va="center",
                    ha="left" if v >= 0 else "right", color=INK2, fontsize=10)
    ax.axvline(0, color=AXIS, linewidth=1, zorder=2)
    ax.set_yticks(y, [label for _, label in methods])
    ax.invert_yaxis()
    ax.set_xlim(-0.36, 0.32)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Pooled skill vs repeating the last value (higher is better)")
    ax.set_title("The bar to beat: Pacific's baselines, reproduced locally")
    ax.legend(loc="lower right", fontsize=9.5)
    save(fig, "baselines.png")


def fig_price_history(files: packlib.PackFiles) -> None:
    s = pd.read_csv(files.series, usecols=["series_id", "date", "value"])
    s["date"] = pd.to_datetime(s["date"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    for sid, color, label in (("aws_spot.p5.48xlarge.az_median", S2, "H100 (p5.48xlarge)"),
                              ("aws_spot.p4d.24xlarge.az_median", S1, "A100 (p4d.24xlarge)")):
        g = s[s["series_id"] == sid].sort_values("date").set_index("date")["value"]
        g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="D"))   # the archive hole stays a gap
        ax.plot(g.index, g.values, color=color, linewidth=2, label=label)
        last = g.dropna()
        ax.text(last.index[-1] + pd.Timedelta(days=12), last.iloc[-1], f"{label.split()[0]} ${last.iloc[-1]:.2f}",
                color=INK2, va="center", fontsize=10)
    ax.axvspan(pd.Timestamp("2026-03-01"), pd.Timestamp("2026-06-30"), color=GRID, alpha=0.6, zorder=0)
    ax.text(pd.Timestamp("2026-04-30"), ax.get_ylim()[1] * 0.95, "archive\nhole", ha="center", va="top", color=MUTED, fontsize=9)
    ax.set_ylabel("USD per GPU-hour")
    ax.set_title("Four years of AWS spot (per-zone median)")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_xlim(right=pd.Timestamp("2027-03-01"))

    ax = axes[1]
    sid = "gpurentalprices.neocloud_rate_card.H100_SXM_80GB.on_demand.median"
    g = s[s["series_id"] == sid].sort_values("date")
    ax.step(g["date"], g["value"], where="post", color=S1, linewidth=2)
    changes = int((g["value"].diff().fillna(0) != 0).sum())
    ax.set_title("A rate card moves in steps")
    ax.set_ylabel("USD per GPU-hour")
    ax.text(0.03, 0.95, f"H100 SXM 80GB, neocloud rate-card median\n(gpurentalprices): {changes} changes in {len(g)} days",
            transform=ax.transAxes, color=INK2, fontsize=9.5, va="top")
    axes[0].xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=(1, 7)))
    axes[1].xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    for a in axes:
        a.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %y"))
    fig.tight_layout()
    save(fig, "price_history.png")


def fig_parity(res: dict) -> None:
    lv, units = res["levels"], res["capability_units"]
    b200 = lv["GIX_B200"]["value"]
    gpus = ["A100", "H100", "H200"]
    now = [lv[f"GIX_{g}"]["value"] for g in gpus]
    parity = [b200 * units[g] / units["B200"] for g in gpus]
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    y = np.arange(len(gpus))
    for yi, a, b in zip(y, parity, now):
        ax.plot([a, b], [yi, yi], color=AXIS, linewidth=2, zorder=2)
    ax.scatter(parity, y, s=90, color=S1, zorder=3, label="if priced like a B200-hour per unit of work", edgecolor=SURFACE, linewidth=2)
    ax.scatter(now, y, s=90, color=S2, zorder=3, label="GIX index, 3 Nov 2026", edgecolor=SURFACE, linewidth=2)
    for yi, a, b in zip(y, parity, now):
        ax.text(b + 0.12, yi, f"${b:.2f}  ({100 * (b / a - 1):+.0f}% above parity)", va="center", color=INK2, fontsize=10)
        ax.text(a - 0.12, yi, f"${a:.2f}", va="center", ha="right", color=INK2, fontsize=10)
    ax.set_yticks(y, gpus)
    ax.set_xlim(0, 7.5)
    ax.set_ylim(-0.6, len(gpus) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("USD per GPU-hour")
    ax.set_title("Older GPUs cost more per unit of work than a B200")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=9.5)
    fig.text(0.01, -0.2, "Capability units: MLPerf inference cross-benchmark median from the pack (H100 = 1, B200 = 3.21); "
             "A100 = 1/2.46 per the brief.", color=MUTED, fontsize=8.5)
    save(fig, "capability_parity.png")


def fig_long_run(res: dict) -> None:
    sc = res["scenarios"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9), sharey=False)
    years = np.linspace(0, 3, 37)
    for ax, gpu, color, band in ((axes[0], "H100", S2, BAND2), (axes[1], "A100", S1, BAND1)):
        s = sc[gpu]
        r0, g = s["rent_now"], s["annual_change"]
        floor = s["floor_usd_per_gpu_hour"]
        path = {k: np.maximum(r0 * (1 + g[k]) ** years, floor[k]) for k in g}
        x = 2026.85 + years
        ax.fill_between(x, path["low"], path["high"], color=band, linewidth=0, label="low to high case")
        ax.plot(x, path["base"], color=color, linewidth=2, label=f"base: market-implied {100 * g['base']:+.0f}%/yr")
        for k in ("low", "high"):
            ax.text(x[-1] + 0.04, path[k][-1], f"{k} ${path[k][-1]:.2f}", va="center", color=MUTED, fontsize=9.5)
        ax.text(x[-1] + 0.04, path["base"][-1], f"base ${path['base'][-1]:.2f}", va="center", color=INK2, fontsize=10)
        ax.scatter([x[0]], [r0], s=60, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(x[0] + 0.08, r0 * 1.03, f"now ${r0:.2f} (GIX)", color=INK2, fontsize=10, va="bottom")
        resale = s["resale_usd"]["3y"]
        ax.set_title(f"{gpu}: rent to Nov 2029; modelled resale ${resale['base'] / 1000:.1f}k "
                     f"(${resale['low'] / 1000:.1f}k–${resale['high'] / 1000:.1f}k)", fontsize=12)
        ax.set_xlim(2026.7, 2030.25)
        ax.set_xticks([2027, 2028, 2029, 2030])
        ax.set_ylim(0, r0 * 1.18)
        ax.set_ylabel("USD per GPU-hour")
        ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    save(fig, "long_run.png")


def main() -> None:
    files = packlib.build(packlib.DATA_CUTOFF)
    res = json.loads((ROOT / "results" / "long_run.json").read_text())
    fig_baselines()
    fig_price_history(files)
    fig_parity(res)
    fig_long_run(res)


if __name__ == "__main__":
    main()
