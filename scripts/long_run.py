"""The 2–3 year view: H100 and A100 rental prices and resale values in late 2028 and 2029.

This cannot be scored by Sunday, so it is an argued, explicit model built only
from the pack (data dated on or before 6 Nov 2026) and stated assumptions:

1. **Where rents are now:** the latest GIX index fixings (one benchmark contract
   per GPU) and the GetDeploying weekly on-demand medians.
2. **How fast they have fallen:** log-linear annual rates from GetDeploying
   (weekly medians since May 2025) and from four years of AWS spot (p4d = A100,
   p5 = H100; per-zone medians, two segments around the archive hole, never
   joined across it).
3. **What newer chips do to older ones:** measured MLPerf capability units
   (H100 = 1). If a B200-hour costs less per unit of work than an H100-hour,
   H100 rents are under pressure to fall to parity.
4. **A floor:** an older GPU is still rented while its rent covers its power.
5. **Resale value = the present value of its remaining rental earnings**, net
   of power and operating cost, at a stated utilisation and discount rate.

Every input is printed with its source; every assumption is a named range.
Instruments are never chained: each rate is reported per instrument and the
scenarios say which one they use. Writes results/long_run.json and
results/long_run.md.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpr import pack as packlib  # noqa: E402

PACK = packlib.PACK_DIR
YEARS_AHEAD = (2.0, 3.0)                      # Nov 2028 and Nov 2029

ASSUMPTIONS = {
    "electricity_usd_per_kwh": (0.06, 0.09, 0.12),     # low / base / high, US industrial range
    "pue": (1.2, 1.3, 1.5),
    "host_overhead": 1.5,                              # server, network, cooling share per GPU watt
    "ops_usd_per_gpu_hour": (0.05, 0.10, 0.15),       # staff, space, maintenance, financing excluded
    "utilisation": (0.85, 0.70, 0.55),                 # high / base / low earning hours share
    "discount_rate": (0.10, 0.13, 0.18),
    "end_of_life": {"A100": 2030.5, "H100": 2032.0},  # rented until about 10 years after launch
}


def log_slope_per_year(dates: pd.Series, values: pd.Series) -> float:
    t = (pd.to_datetime(dates) - pd.Timestamp("2020-01-01")).dt.days.to_numpy() / 365.25
    v = values.to_numpy(dtype=float)
    ok = np.isfinite(v) & (v > 0)
    if ok.sum() < 8:
        return float("nan")
    slope = np.polyfit(t[ok], np.log(v[ok]), 1)[0]
    return float(math.exp(slope) - 1)


def current_levels() -> dict:
    ix = pd.read_csv(PACK / "index_prints.csv")
    ix = ix[ix["status"].astype(str).str.lower().ne("withheld")]
    out = {}
    for gpu in ("A100", "H100", "H200", "B200", "B300"):
        g = ix[ix["index_id"] == f"GIX_{gpu}"].dropna(subset=["value_usd_per_gpu_hour"]).sort_values("print_date")
        if len(g):
            out[f"GIX_{gpu}"] = {"value": float(g["value_usd_per_gpu_hour"].iloc[-1]), "date": g["print_date"].iloc[-1]}
    ag = pd.read_csv(PACK / "aggregator_medians.csv")
    for gpu in ("A100", "H100", "H200", "B200"):
        g = ag[(ag["gpu_model"] == f"NVIDIA {gpu}")]
        for mode in ("on_demand", "reserved_12m", "reserved_36m", "spot"):
            m = g[g["pricing_mode"] == mode].sort_values("week_start")
            final = m[m["status"].astype(str) != "provisional"] if "status" in m else m
            if len(final):
                out[f"GetDeploying_{gpu}_{mode}"] = {"value": float(final["median_usd_per_gpu_hour"].iloc[-1]),
                                                     "date": final["week_start"].iloc[-1]}
    return out


def trends() -> dict:
    out = {}
    ag = pd.read_csv(PACK / "aggregator_medians.csv")
    for gpu in ("A100", "H100", "H200", "B200"):
        m = ag[(ag["gpu_model"] == f"NVIDIA {gpu}") & (ag["pricing_mode"] == "on_demand")]
        m = m[m["status"].astype(str) != "provisional"]
        if len(m):
            out[f"GetDeploying_{gpu}_on_demand"] = {
                "annual_change": log_slope_per_year(m["week_start"], m["median_usd_per_gpu_hour"]),
                "from": m["week_start"].min(), "to": m["week_start"].max(), "n": len(m)}
    files = packlib.build(packlib.DATA_CUTOFF)
    series = pd.read_csv(files.series, usecols=["series_id", "date", "value"])
    for gpu, inst in (("A100", "p4d.24xlarge"), ("H100", "p5.48xlarge")):
        s = series[series["series_id"] == f"aws_spot.{inst}.az_median"].sort_values("date")
        if s.empty:
            continue
        seg1 = s[s["date"] <= "2026-02-28"]
        seg2 = s[s["date"] >= "2026-07-01"]
        out[f"AWS_spot_{gpu}_{inst}"] = {
            "annual_change_segment_1": log_slope_per_year(seg1["date"], seg1["value"]),
            "segment_1": [seg1["date"].min(), seg1["date"].max()],
            "first_value": float(seg1["value"].iloc[0]) if len(seg1) else None,
            "last_value_before_hole": float(seg1["value"].iloc[-1]) if len(seg1) else None,
            "last_value": float(s["value"].iloc[-1]), "last_date": s["date"].iloc[-1]}
    return out


def capability() -> dict:
    cu = pd.read_csv(PACK / "capability" / "capability_units.csv")
    m = cu[cu["benchmark_set_id"] == "ship_only_inf_cross_benchmark_median_h100ref_sxm80"]
    units = {r.gpu_generation: float(r.capability_units_per_accelerator) for r in m.itertuples()}
    units["A100"] = 1 / 2.46   # the brief: H100 = 2.46x A100, Inference v3.0/v3.1 cross-benchmark median
    return units


def launch_prices() -> dict:
    hf = pd.read_csv(PACK / "hardware_fundamentals" / "hardware_fundamentals.csv")
    want = {"H100": "NVIDIA H100 SXM5 80GB", "A100": "NVIDIA A100 SXM4 40 GB", "H200": "NVIDIA H200 SXM"}
    out = {}
    for gpu, name in want.items():
        row = hf[hf["accelerator"] == name]
        if len(row):
            out[gpu] = {"release_price_usd": float(row["release_price_usd"].iloc[0]), "tdp_w": float(row["tdp_w"].iloc[0]),
                        "release_date": row["release_date"].iloc[0], "accelerator": name}
    return out


def power_cost(tdp_w: float, k: int) -> float:
    a = ASSUMPTIONS
    return tdp_w / 1000 * a["host_overhead"] * a["pue"][k] * a["electricity_usd_per_kwh"][k]


def implied_decline(ratio: float, years: float = 3.0) -> float:
    """Annual change g at which the average rent over ``years`` equals ``ratio`` x today's rent.

    A lessor quoting a 36-month price below on-demand is, roughly, pricing the
    average rent it expects over the term (ignoring the utilisation and credit
    risk the reservation removes, which push the same way). Solve
    ``(e^(k T) - 1) / (k T) = ratio`` for ``k``, then ``g = e^k - 1``.
    """
    if not (0 < ratio < 1.5):
        return float("nan")
    lo, hi = -3.0, 1.0
    for _ in range(100):
        k = (lo + hi) / 2
        avg = 1.0 if abs(k) < 1e-9 else (math.exp(k * years) - 1) / (k * years)
        lo, hi = (k, hi) if avg < ratio else (lo, k)
    return math.exp((lo + hi) / 2) - 1


def scenarios(levels: dict, tr: dict, units: dict, launch: dict) -> dict:
    """Three cases per GPU, each a constant annual rent change from today's index level:

    - **low:** -35%/yr, about the fall of AWS H100 spot after launch (the fastest observed decline);
    - **base:** the decline implied by GetDeploying's 36-month reserved price against on-demand;
    - **high:** -4%/yr, A100 AWS spot over four years (the slowest observed decline).

    The recent rise in GetDeploying on-demand medians is reported as upside risk, not
    extrapolated: a pooled median moves with which listings it holds, and a rise in an
    ageing chip's rent is not a 3-year trend.
    """
    out = {}
    tdp = {"H100": 700.0, "A100": 400.0}
    for gpu in ("H100", "A100"):
        r0 = levels.get(f"GIX_{gpu}", {}).get("value") or levels.get(f"GetDeploying_{gpu}_on_demand", {}).get("value")
        od = levels.get(f"GetDeploying_{gpu}_on_demand", {}).get("value")
        r36 = levels.get(f"GetDeploying_{gpu}_reserved_36m", {}).get("value")
        implied = implied_decline(r36 / od) if od and r36 else float("nan")
        b200 = levels.get("GIX_B200", {}).get("value")
        parity = b200 * units[gpu] / units["B200"] if b200 and gpu in units and "B200" in units else None
        named = {"low": -0.35, "base": implied if math.isfinite(implied) else -0.15, "high": -0.04}
        floor = {k: power_cost(tdp[gpu], i) + ASSUMPTIONS["ops_usd_per_gpu_hour"][i]
                 for i, k in enumerate(("low", "base", "high"))}
        rent = {f"{y:g}y": {k: max(r0 * (1 + named[k]) ** y, floor[k]) for k in named} for y in YEARS_AHEAD}
        resale = {}
        for years in YEARS_AHEAD:
            start = 2026.85 + years
            life = max(0.0, ASSUMPTIONS["end_of_life"][gpu] - start)
            row = {}
            for k, j in (("low", 2), ("base", 1), ("high", 0)):   # low case: low utilisation, high cost, high discount
                util = ASSUMPTIONS["utilisation"][{"low": 2, "base": 1, "high": 0}[k]]
                disc = ASSUMPTIONS["discount_rate"][j]
                cost = power_cost(tdp[gpu], j) + ASSUMPTIONS["ops_usd_per_gpu_hour"][j]
                pv, t = 0.0, 0.0
                while t < life - 1e-9:
                    step = min(1.0, life - t)
                    r = r0 * (1 + named[k]) ** (years + t)
                    pv += max(0.0, util * r - cost) * 8760 * step / (1 + disc) ** (t + step / 2)
                    t += step
                row[k] = pv
            price = launch.get(gpu, {}).get("release_price_usd")
            resale[f"{years:g}y"] = {**row, "remaining_life_years": life,
                                     **({"share_of_launch_price": {k: v / price for k, v in row.items()}} if price else {})}
        out[gpu] = {"rent_now": r0, "annual_change": named, "implied_by_36m_vs_on_demand": implied,
                    "reserved_36m_over_on_demand": (r36 / od) if od and r36 else None,
                    "capability_parity_rent_now": parity, "parity_gap": (r0 / parity - 1) if parity else None,
                    "floor_usd_per_gpu_hour": floor, "rent": rent, "resale_usd": resale}
    return out


def to_markdown(res: dict) -> str:
    lv, tr, sc, units = res["levels"], res["trends"], res["scenarios"], res["capability_units"]
    lines = ["# The 2–3 year view (argued, not scored)", "",
             "Built only from Pacific's pack (data to 6 Nov 2026) and the stated assumptions; "
             "`python scripts/long_run.py` regenerates it.", "", "## Evidence", "",
             "| Input | Value | Date / span | Instrument |", "|---|---|---|---|"]
    for k, v in lv.items():
        lines.append(f"| {k} | ${v['value']:.2f}/GPU-h | {v['date']} | "
                     f"{'standardized_index' if k.startswith('GIX') else 'aggregator_median'} |")
    for k, v in tr.items():
        if "annual_change" in v:
            lines.append(f"| {k} trend | {100 * v['annual_change']:+.0f}%/yr | {v['from']} to {v['to']} | aggregator_median |")
        else:
            lines.append(f"| {k} trend (before the archive hole) | {100 * v['annual_change_segment_1']:+.0f}%/yr | "
                         f"{v['segment_1'][0]} to {v['segment_1'][1]} | hyperscaler_spot |")
    lines.append("| Capability units (H100 = 1, MLPerf inference cross-benchmark median) | "
                 + ", ".join(f"{g} {u:.2f}" for g, u in sorted(units.items(), key=lambda x: x[1])) + " | Inference v3.0–v6.1 | benchmark_measured |")
    lines += ["", "## Scenarios", "",
              "Each case is a constant annual rent change from today's GIX level: **low** -35%/yr (AWS H100 spot after launch), "
              "**base** the decline implied by GetDeploying's 36-month reserved price against on-demand, **high** -4%/yr "
              "(A100 AWS spot over four years). Rents never fall below a power-plus-operations floor. Resale is the present "
              "value of the remaining net earnings to end of life, with utilisation, costs and discount rate moving with the case:",
              "", f"`{json.dumps(ASSUMPTIONS)}`", ""]
    for gpu, s in sc.items():
        lines += [f"### {gpu}", "",
                  f"- Rent now: ${s['rent_now']:.2f}/GPU-h (GIX). At B200 parity per unit of work it would be "
                  + (f"${s['capability_parity_rent_now']:.2f}, so it is priced {100 * s['parity_gap']:+.0f}% above parity"
                     if s['capability_parity_rent_now'] else "n/a") + ".",
                  f"- 36-month reserved / on-demand (GetDeploying): {s['reserved_36m_over_on_demand']:.2f}, implying "
                  f"{100 * s['implied_by_36m_vs_on_demand']:+.0f}%/yr.",
                  "- Annual change used: " + ", ".join(f"{k} {100 * v:+.0f}%" for k, v in s["annual_change"].items()),
                  "- Floor (power and operations): " + ", ".join(f"{k} ${v:.2f}" for k, v in s["floor_usd_per_gpu_hour"].items()),
                  "", "| Horizon | Rent low | Rent base | Rent high | Resale low | Resale base | Resale high |",
                  "|---|---|---|---|---|---|---|"]
        for h in s["rent"]:
            r, v = s["rent"][h], s["resale_usd"][h]
            lines.append(f"| {h} | ${r['low']:.2f} | ${r['base']:.2f} | ${r['high']:.2f} | "
                         f"${v['low']:,.0f} | ${v['base']:,.0f} | ${v['high']:,.0f} |")
        share = s["resale_usd"]["3y"].get("share_of_launch_price")
        if share:
            lines.append("Resale in 3 years as a share of the launch price: "
                         + ", ".join(f"{k} {100 * v:.0f}%" for k, v in share.items()) + ".")
        lines.append("")
    lines += ["Resale is a modelled present value of remaining earnings, not a quoted price: no open resale dataset "
              "exists. Compare it with the publishers in the pack's `resale_pointers.csv` (cited, values not stored).", ""]
    return "\n".join(lines)


def main() -> None:
    res = {"levels": current_levels(), "trends": trends(), "capability_units": capability(), "launch": launch_prices()}
    res["scenarios"] = scenarios(res["levels"], res["trends"], res["capability_units"], res["launch"])
    res["assumptions"] = ASSUMPTIONS
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "long_run.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "long_run.md").write_text(to_markdown(res))
    print(to_markdown(res))


if __name__ == "__main__":
    main()
