"""Prepare the public Stanford/Detroit longitudinal SomaLogic PE cohorts.

The Figshare workbook is immutable. Assays are matched by SomaId, which is more
reliable than splitting multi-gene display labels. Stanford assay columns are
named with the exact feature names used by ``PE_final.csv`` because that matrix
is a relabelled version of the Stanford sheet (98 visits, 36 participants).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "external" / "raw" / "figshare_7962998_registry.xlsx"
OUT = ROOT / "data" / "external" / "processed"
DISCOVERY = ROOT / "data" / "processed" / "PE_final.csv"


def read_clinical(sheet: str, cohort: str) -> pd.DataFrame:
    frame = pd.read_excel(SOURCE, sheet_name=sheet)
    frame.columns = [str(column).strip() for column in frame.columns]
    frame["ID"] = cohort + "_" + frame.ID.astype(int).astype(str)
    frame.insert(0, "Cohort", cohort)
    frame = frame.rename(columns={"Group": "ClinicalGroup", "Age": "MaternalAge", "Parity": "Parity"})
    return frame


def read_proteomics(sheet: str, cohort: str, soma_to_feature: dict[str, str] | None = None):
    raw = pd.read_excel(SOURCE, sheet_name=sheet, header=None)
    soma_ids = raw.iloc[0, 3:].astype(str).str.strip().tolist()
    values = raw.iloc[8:, 3:].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    metadata = raw.iloc[8:, :3].copy().reset_index(drop=True)
    metadata.columns = ["ID", "GestationalAge", "Phenotype"]
    metadata["ID"] = cohort + "_" + metadata.ID.astype(int).astype(str)
    metadata.insert(0, "Cohort", cohort)
    metadata["Label"] = metadata.Phenotype.astype(str).str.startswith("PE").astype(int)
    metadata["LatePE"] = metadata.Phenotype.astype(str).isin(["PE-late", "PE"]).astype(int)

    if soma_to_feature is None:
        discovery_features = list(pd.read_csv(DISCOVERY, nrows=0).columns[3:])
        if len(discovery_features) != len(soma_ids):
            raise ValueError("Stanford assay count does not match PE_final feature count")
        feature_names = discovery_features
        soma_to_feature = dict(zip(soma_ids, feature_names))
        keep = list(range(len(soma_ids)))
    else:
        keep = [index for index, soma_id in enumerate(soma_ids) if soma_id in soma_to_feature]
        feature_names = [soma_to_feature[soma_ids[index]] for index in keep]

    proteins = values.iloc[:, keep].copy()
    proteins.columns = feature_names
    if proteins.columns.duplicated().any():
        raise ValueError("SomaId mapping produced duplicate feature names")
    return pd.concat([metadata, proteins], axis=1), soma_to_feature, len(soma_ids)


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    OUT.mkdir(parents=True, exist_ok=True)

    stanford, mapping, stanford_assays = read_proteomics("Proteomic data Stanford", "Stanford")
    detroit, _, detroit_assays = read_proteomics("Proteomic data Detroit", "Detroit", mapping)
    stanford_clinical = read_clinical("Patient data Stanford", "Stanford")
    detroit_clinical = read_clinical("Patient data Detroit", "Detroit")

    stanford = stanford.merge(stanford_clinical.drop(columns=["Cohort"]), on="ID", how="left")
    detroit = detroit.merge(detroit_clinical.drop(columns=["Cohort"]), on="ID", how="left")

    paths = {
        "stanford": OUT / "Stanford_SomaLogic_longitudinal_PE.csv",
        "detroit": OUT / "Detroit_SomaLogic_longitudinal_PE.csv",
        "stanford_clinical": OUT / "Stanford_clinical.csv",
        "detroit_clinical": OUT / "Detroit_clinical.csv",
    }
    stanford.to_csv(paths["stanford"], index=False)
    detroit.to_csv(paths["detroit"], index=False)
    stanford_clinical.to_csv(paths["stanford_clinical"], index=False)
    detroit_clinical.to_csv(paths["detroit_clinical"], index=False)

    report = {
        "source": "https://doi.org/10.6084/m9.figshare.7962998.v1",
        "license": "CC BY 4.0",
        "source_md5": "cb1c63554efb1b04f5b1b6cd6b5f3db7",
        "stanford_visits": int(len(stanford)),
        "stanford_participants": int(stanford.ID.nunique()),
        "stanford_late_pe": int(stanford.loc[stanford.LatePE == 1, "ID"].nunique()),
        "stanford_early_pe": int(stanford.loc[stanford.Phenotype == "PE-early", "ID"].nunique()),
        "stanford_controls": int(stanford.loc[stanford.Phenotype == "Term", "ID"].nunique()),
        "detroit_visits": int(len(detroit)),
        "detroit_participants": int(detroit.ID.nunique()),
        "detroit_late_pe": int(detroit.loc[detroit.LatePE == 1, "ID"].nunique()),
        "detroit_controls": int(detroit.loc[detroit.Phenotype == "Control", "ID"].nunique()),
        "stanford_assays": int(stanford_assays),
        "detroit_assays": int(detroit_assays),
        "shared_somaid_features": int(detroit.shape[1] - 12),
    }
    (OUT / "figshare_multicohort_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
