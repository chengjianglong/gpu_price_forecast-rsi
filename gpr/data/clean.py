"""Turn price quotes extracted from web pages into clean per-GPU observations.

Extraction is done by Exa's structured summaries, so quotes can be wrong:
the wrong GPU, a node price read as a per-GPU price, a monthly price read as
hourly. Everything here is a cheap rule that rejects or fixes those cases.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

# Plausible per-GPU hourly prices in USD. Wide on purpose: they only catch
# extraction errors (node prices, monthly prices, typos), not real moves.
PRICE_BOUNDS = {
    "A100": (0.2, 8.0),
    "H100": (0.5, 15.0),
    "H200": (0.8, 18.0),
    "B200": (1.0, 25.0),
}

_GPU_PATTERNS = [
    ("B200", re.compile(r"\bB\s?200\b", re.I)),
    ("H200", re.compile(r"\bH\s?200\b", re.I)),
    ("H100", re.compile(r"\bH\s?100\b", re.I)),
    ("A100", re.compile(r"\bA\s?100\b", re.I)),
]

_PRICING_ALIASES = {
    "on_demand": ("on-demand", "on demand", "ondemand", "on_demand", "pay as you go", "pay-as-you-go", "hourly"),
    "spot": ("spot", "interruptible", "preemptible", "community", "bid"),
    "reserved": ("reserved", "commit", "contract", "1-year", "1 year", "3-year", "3 year", "annual", "monthly plan"),
}

# Multi-part public suffixes we see on provider sites.
_TWO_PART_TLDS = {"co.uk", "com.au", "co.jp"}


def normalise_gpu(name: str | None) -> tuple[str | None, str]:
    """Return (canonical GPU, variant text) or (None, "") if not a tracked GPU.

    "NVIDIA H100 SXM5 80GB" -> ("H100", "SXM5 80GB"). GH200 is a different
    product, so it is not counted as H200.
    """
    text = str(name or "")
    if re.search(r"\bGH\s?200\b", text, re.I):
        return None, ""
    for gpu, pattern in _GPU_PATTERNS:
        if pattern.search(text):
            variant = pattern.sub("", text)
            variant = re.sub(r"\bNVIDIA\b", "", variant, flags=re.I)
            return gpu, " ".join(variant.split())
    return None, ""


def normalise_pricing_type(value: str | None) -> str:
    text = str(value or "").strip().lower()
    for canonical, aliases in _PRICING_ALIASES.items():
        if any(alias in text for alias in aliases):
            return canonical
    return "on_demand"  # list prices on pricing pages are on-demand unless stated


def provider_from_url(url: str) -> str:
    """"https://www.lambda.ai/pricing" -> "lambda"."""
    host = (urlparse(url).hostname or "").lower()
    parts = [p for p in host.split(".") if p and p != "www"]
    if len(parts) >= 3 and ".".join(parts[-2:]) in _TWO_PART_TLDS:
        return parts[-3]
    return parts[-2] if len(parts) >= 2 else (parts[0] if parts else "unknown")


def per_gpu_hourly(quote: dict[str, Any]) -> float | None:
    """Convert a quote to USD per GPU-hour, or None if it can't be trusted."""
    try:
        price = float(quote.get("price_usd"))
    except (TypeError, ValueError):
        return None
    if price <= 0:
        return None
    unit = str(quote.get("price_unit") or "per_gpu_hour").lower()
    try:
        gpus = max(1, int(quote.get("gpus_per_instance") or 1))
    except (TypeError, ValueError):
        gpus = 1
    if "month" in unit:
        price /= 730.0  # average hours per month
    if "instance" in unit or "node" in unit:
        price /= gpus
    return price


def clean_quote(quote: dict[str, Any]) -> dict[str, Any] | None:
    """Return a normalised quote, or None if it fails a check."""
    gpu, variant = normalise_gpu(quote.get("gpu"))
    if gpu is None:
        return None
    price = per_gpu_hourly(quote)
    if price is None:
        return None
    low, high = PRICE_BOUNDS[gpu]
    if not (low <= price <= high):
        return None
    try:
        gpus = max(1, int(quote.get("gpus_per_instance") or 1))
    except (TypeError, ValueError):
        gpus = 1
    return {
        "gpu": gpu,
        "variant": variant,
        "pricing_type": normalise_pricing_type(quote.get("pricing_type")),
        "usd_per_gpu_hour": round(price, 4),
        "gpus_per_instance": gpus,
    }
