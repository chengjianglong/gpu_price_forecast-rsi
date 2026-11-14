# The 2–3 year view (argued, not scored)

Built only from Pacific's pack (data to 6 Nov 2026) and the stated assumptions; `python scripts/long_run.py` regenerates it.

## Evidence

| Input | Value | Date / span | Instrument |
|---|---|---|---|
| GIX_A100 | $1.98/GPU-h | 2026-11-03 | standardized_index |
| GIX_H100 | $3.33/GPU-h | 2026-11-03 | standardized_index |
| GIX_H200 | $4.28/GPU-h | 2026-11-03 | standardized_index |
| GIX_B200 | $7.32/GPU-h | 2026-11-03 | standardized_index |
| GetDeploying_A100_on_demand | $1.79/GPU-h | 2026-10-26 | aggregator_median |
| GetDeploying_A100_reserved_12m | $1.70/GPU-h | 2026-10-26 | aggregator_median |
| GetDeploying_A100_reserved_36m | $1.36/GPU-h | 2026-10-26 | aggregator_median |
| GetDeploying_A100_spot | $1.36/GPU-h | 2026-10-26 | aggregator_median |
| GetDeploying_H100_on_demand | $3.85/GPU-h | 2026-10-26 | aggregator_median |
| GetDeploying_H100_reserved_12m | $3.21/GPU-h | 2026-10-26 | aggregator_median |
| GetDeploying_H100_reserved_36m | $2.97/GPU-h | 2026-10-26 | aggregator_median |
| GetDeploying_H100_spot | $2.39/GPU-h | 2026-10-26 | aggregator_median |
| GetDeploying_H200_on_demand | $4.81/GPU-h | 2026-10-26 | aggregator_median |
| GetDeploying_H200_reserved_12m | $3.56/GPU-h | 2026-10-26 | aggregator_median |
| GetDeploying_H200_reserved_36m | $2.97/GPU-h | 2026-10-26 | aggregator_median |
| GetDeploying_H200_spot | $3.57/GPU-h | 2026-10-26 | aggregator_median |
| GetDeploying_B200_on_demand | $7.83/GPU-h | 2026-10-26 | aggregator_median |
| GetDeploying_B200_reserved_12m | $8.68/GPU-h | 2026-10-26 | aggregator_median |
| GetDeploying_B200_reserved_36m | $7.16/GPU-h | 2026-10-26 | aggregator_median |
| GetDeploying_B200_spot | $4.78/GPU-h | 2026-10-26 | aggregator_median |
| GetDeploying_A100_on_demand trend | +15%/yr | 2025-11-03 to 2026-10-26 | aggregator_median |
| GetDeploying_H100_on_demand trend | +26%/yr | 2025-11-03 to 2026-10-26 | aggregator_median |
| GetDeploying_H200_on_demand trend | +37%/yr | 2025-11-03 to 2026-10-26 | aggregator_median |
| GetDeploying_B200_on_demand trend | +45%/yr | 2025-11-03 to 2026-10-26 | aggregator_median |
| AWS_spot_A100_p4d.24xlarge trend (before the archive hole) | -4%/yr | 2022-06-01 to 2026-02-28 | hyperscaler_spot |
| AWS_spot_H100_p5.48xlarge trend (before the archive hole) | -33%/yr | 2024-12-19 to 2026-02-28 | hyperscaler_spot |
| Capability units (H100 = 1, MLPerf inference cross-benchmark median) | A100 0.41, H100 1.00, H200 1.10, B200 3.21, GB200 3.28, B300 3.56, GB300 4.13 | Inference v3.0–v6.1 | benchmark_measured |

## Scenarios

Each case is a constant annual rent change from today's GIX level: **low** -35%/yr (AWS H100 spot after launch), **base** the decline implied by GetDeploying's 36-month reserved price against on-demand, **high** -4%/yr (A100 AWS spot over four years). Rents never fall below a power-plus-operations floor. Resale is the present value of the remaining net earnings to end of life, with utilisation, costs and discount rate moving with the case:

`{"electricity_usd_per_kwh": [0.06, 0.09, 0.12], "pue": [1.2, 1.3, 1.5], "host_overhead": 1.5, "ops_usd_per_gpu_hour": [0.05, 0.1, 0.15], "utilisation": [0.85, 0.7, 0.55], "discount_rate": [0.1, 0.13, 0.18], "end_of_life": {"A100": 2030.5, "H100": 2032.0}}`

### H100

- Rent now: $3.33/GPU-h (GIX). At B200 parity per unit of work it would be $2.28, so it is priced +46% above parity.
- 36-month reserved / on-demand (GetDeploying): 0.77, implying -17%/yr.
- Annual change used: low -35%, base -17%, high -4%
- Floor (power and operations): low $0.13, base $0.22, high $0.34

| Horizon | Rent low | Rent base | Rent high | Resale low | Resale base | Resale high |
|---|---|---|---|---|---|---|
| 2y | $1.41 | $2.32 | $3.07 | $4,640 | $26,369 | $56,748 |
| 3y | $0.92 | $1.94 | $2.95 | $1,328 | $16,727 | $39,577 |
Resale in 3 years as a share of the launch price: low 4%, base 50%, high 118%.

### A100

- Rent now: $1.98/GPU-h (GIX). At B200 parity per unit of work it would be $0.93, so it is priced +114% above parity.
- 36-month reserved / on-demand (GetDeploying): 0.76, implying -17%/yr.
- Annual change used: low -35%, base -17%, high -4%
- Floor (power and operations): low $0.09, base $0.17, high $0.26

| Horizon | Rent low | Rent base | Rent high | Resale low | Resale base | Resale high |
|---|---|---|---|---|---|---|
| 2y | $0.84 | $1.35 | $1.83 | $1,822 | $9,372 | $19,201 |
| 3y | $0.54 | $1.12 | $1.75 | $223 | $3,353 | $7,713 |
Resale in 3 years as a share of the launch price: low 1%, base 22%, high 51%.

Resale is a modelled present value of remaining earnings, not a quoted price: no open resale dataset exists. Compare it with the publishers in the pack's `resale_pointers.csv` (cited, values not stored).
