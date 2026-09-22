"""Conservative display-label normalization for exploratory biology only.

Primary Stanford/Detroit assay matching uses SomaId and never calls this helper.
"""

from __future__ import annotations

import re


def canonical_symbol(value: str) -> str:
    value = str(value).strip().upper()
    value = re.sub(r"\.\d+$", "", value)
    value = {
        "PLGF": "PGF", "MMP-7": "MMP7", "HMG-1": "HMGB1",
        "TIMP-1": "TIMP1", "TIMP-2": "TIMP2", "TIMP-3": "TIMP3",
    }.get(value, value)
    if "/" in value:
        value = value.rsplit("/", 1)[-1]
    elif "." in value:
        pieces = [piece for piece in value.split(".") if piece]
        value = pieces[-1] if pieces else value
    elif "," in value:
        value = value.split(",", 1)[0]
    elif " " in value:
        return ""
    value = re.sub(r"[^A-Z0-9-]", "", value)
    return value if re.fullmatch(r"[A-Z][A-Z0-9-]{1,14}", value) else ""
