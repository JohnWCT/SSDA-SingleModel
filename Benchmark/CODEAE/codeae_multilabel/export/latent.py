"""Sample-level latent export after fine-tuning."""

from __future__ import annotations

import pandas as pd
import torch

from codeae_multilabel.contracts import OmicsTable
from codeae_multilabel.model.wrapper import MultiLabelCodeAEModel


def extract_latent_table(
    model: MultiLabelCodeAEModel,
    omics: OmicsTable,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    model.eval()
    model.to(device)
    x = omics.x.loc[list(omics.sample_ids)].values.astype("float32")
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch_x = torch.as_tensor(
                x[start : start + batch_size], dtype=torch.float32, device=device
            )
            z = model.encode(batch_x, deterministic=True)
            z_np = z.cpu().numpy()
            for i, sid in enumerate(omics.sample_ids[start : start + batch_size]):
                row: dict[str, object] = {"sample_id": sid, "domain": omics.domain}
                for j in range(z_np.shape[1]):
                    row[f"latent_{j}"] = float(z_np[i, j])
                rows.append(row)
    return pd.DataFrame(rows)


def latent_table_to_dict(latent_df: pd.DataFrame) -> dict[str, list[float]]:
    latent_cols = [c for c in latent_df.columns if c.startswith("latent_")]
    out: dict[str, list[float]] = {}
    for _, row in latent_df.iterrows():
        sid = str(row["sample_id"])
        out[sid] = [float(row[c]) for c in latent_cols]
    return out
