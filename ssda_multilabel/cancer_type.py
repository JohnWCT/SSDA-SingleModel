"""Cancer type metadata alignment for multi-label pipeline."""

from __future__ import annotations

import pandas as pd

UNKNOWN = "Unknown"


def load_and_align_cancer_types(
    source_ids: list[str],
    target_ids: list[str],
    source_path: str | None,
    target_path: str | None,
    sample_id_col: str,
    cancer_type_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, str]] = []

    def _load(path: str | None, domain: str, ids: list[str]) -> None:
        if path is None:
            for sid in ids:
                rows.append({"sample_id": sid, "domain": domain, "cancer_type": UNKNOWN})
            return
        df = pd.read_csv(path)
        sid_c = sample_id_col if sample_id_col in df.columns else "sample_id"
        ct_c = cancer_type_col if cancer_type_col in df.columns else "cancer_type"
        mapping = dict(zip(df[sid_c].astype(str).str.strip(), df[ct_c].astype(str)))
        for sid in ids:
            rows.append(
                {
                    "sample_id": sid,
                    "domain": domain,
                    "cancer_type": mapping.get(sid, UNKNOWN),
                }
            )

    _load(source_path, "source", source_ids)
    _load(target_path, "target", target_ids)
    full = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "domain": d,
                "n_samples": int((full["domain"] == d).sum()),
                "n_unknown": int(((full["domain"] == d) & (full["cancer_type"] == UNKNOWN)).sum()),
            }
            for d in ("source", "target")
        ]
    )
    return full, summary
