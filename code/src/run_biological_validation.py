"""Biological and longitudinal validation of reproducibly selected PE proteins."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, mannwhitneyu

from protein_symbols import canonical_symbol


ROOT = Path(__file__).resolve().parents[1]
SELECTIONS = ROOT / "results" / "main_benchmark" / "benchmark_with_new" / "selected_features.csv"
PE = ROOT / "data" / "processed" / "PE_final.csv"
REACTOME = ROOT / "data" / "external" / "processed" / "ReactomePathways.gmt"
OUT = ROOT / "results" / "biological_validation"
OUT.mkdir(parents=True, exist_ok=True)

# Markers reported in the two independent longitudinal PE proteomics studies
# used for contextual validation (Erez et al. 2017; Tarca et al. 2022).
KNOWN_PE_MARKERS = {
    "ADAM12", "BMP1", "BTK", "CAMK2A", "CAMK2B", "CAMK2D", "CD86", "CDH5",
    "CDK8", "CNTF", "DPT", "F3", "FCN2", "FYN", "HMGB1", "HSPA1A", "ICAM5",
    "IGF1", "INHBA", "ITGA2B", "ITGB3", "KDR", "KLKB1", "KYNU", "METAP1",
    "MMP7", "NAGK", "NID1", "PDE7A", "PDPK1", "PGF", "PPID", "RAN", "SAA1",
    "SERPING1", "SIGLEC6", "TEC", "TF", "TIE1", "TNFAIP6", "VEGFA", "XPNPEP1",
}


def bh(pvalues):
    values = np.asarray(pvalues, float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def stable_panels(selections: pd.DataFrame, top_n: int = 20):
    total_runs = selections[["Repeat", "Fold"]].drop_duplicates().shape[0]
    frequency = (
        selections.drop_duplicates(["Repeat", "Fold", "Method", "Feature"])
        .groupby(["Method", "Feature"], as_index=False).size()
        .rename(columns={"size": "SelectedRuns"})
    )
    frequency["SelectionFrequency"] = frequency.SelectedRuns / total_runs
    frequency["CanonicalFeature"] = frequency.Feature.map(canonical_symbol)
    panels = {
        method: set(group.sort_values(["SelectionFrequency", "Feature"], ascending=[False, True]).head(top_n).CanonicalFeature)
        for method, group in frequency.groupby("Method")
    }
    return frequency, panels


def known_marker_enrichment(panels, universe):
    known = KNOWN_PE_MARKERS & universe
    rows = []
    for method, panel in panels.items():
        panel = panel & universe
        overlap = sorted(panel & known)
        pvalue = hypergeom.sf(len(overlap) - 1, len(universe), len(known), len(panel)) if panel else 1.0
        rows.append({
            "Method": method, "PanelSize": len(panel), "KnownMarkerUniverse": len(known),
            "OverlapCount": len(overlap), "Overlap": ";".join(overlap), "HypergeometricP": pvalue,
        })
    table = pd.DataFrame(rows)
    table["BHAdjustedP"] = bh(table.HypergeometricP)
    return table


def read_reactome():
    pathways = {}
    with REACTOME.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 4:
                pathways[(fields[0], fields[1])] = {canonical_symbol(gene) for gene in fields[2:]}
    return pathways


def reactome_enrichment(panels, universe):
    pathways = read_reactome()
    rows = []
    for method, panel in panels.items():
        panel = panel & universe
        for (pathway, identifier), genes in pathways.items():
            genes &= universe
            if not 10 <= len(genes) <= 500:
                continue
            overlap = sorted(panel & genes)
            if len(overlap) < 2:
                continue
            pvalue = hypergeom.sf(len(overlap) - 1, len(universe), len(genes), len(panel))
            rows.append({
                "Method": method, "Pathway": pathway, "ReactomeID": identifier,
                "PanelSize": len(panel), "PathwaySize": len(genes),
                "OverlapCount": len(overlap), "Overlap": ";".join(overlap),
                "HypergeometricP": pvalue,
            })
    table = pd.DataFrame(rows)
    if not table.empty:
        table["BHAdjustedP"] = table.groupby("Method")["HypergeometricP"].transform(lambda x: bh(x.to_numpy()))
        table = table.sort_values(["Method", "BHAdjustedP", "HypergeometricP"])
    return table


def time_adjusted_associations(features):
    frame = pd.read_csv(PE)
    time = frame.time.map({"1st Tri": 1.0, "2nd Tri": 2.0, "3rd Tri": 3.0}).to_numpy(float)
    label = (frame.label.astype(str) == "PE").astype(int)
    available = {canonical_symbol(column): column for column in frame.columns}
    rows = []
    for feature in sorted(set(features) & set(available)):
        values = pd.to_numeric(frame[available[feature]], errors="coerce").to_numpy(float)
        valid = np.isfinite(values)
        design = np.column_stack([np.ones(valid.sum()), time[valid], time[valid] ** 2])
        coefficients = np.linalg.lstsq(design, values[valid], rcond=None)[0]
        residual = np.full(len(values), np.nan)
        residual[valid] = values[valid] - design @ coefficients
        subject = pd.DataFrame({"ID": frame.ID, "label": label, "residual": residual}).groupby("ID", as_index=False).agg(
            label=("label", "first"), residual=("residual", "mean")
        ).dropna()
        cases = subject.loc[subject.label == 1, "residual"]
        controls = subject.loc[subject.label == 0, "residual"]
        if len(cases) and len(controls):
            statistic, pvalue = mannwhitneyu(cases, controls, alternative="two-sided")
            effect = cases.median() - controls.median()
            rows.append({
                "Feature": feature, "PEParticipants": len(cases), "ControlParticipants": len(controls),
                "TimeAdjustedMedianDifference": effect, "MannWhitneyU": statistic, "PValue": pvalue,
            })
    table = pd.DataFrame(rows)
    if not table.empty:
        table["BHAdjustedP"] = bh(table.PValue)
        table = table.sort_values("BHAdjustedP")
    return table


def pathway_edges(enrichment: pd.DataFrame):
    rows = []
    if enrichment.empty:
        return pd.DataFrame(columns=["Method", "ProteinA", "ProteinB", "SharedPathways"])
    for method, subset in enrichment.groupby("Method"):
        counts = {}
        for genes in subset.Overlap:
            items = sorted(set(genes.split(";")))
            for i, left in enumerate(items):
                for right in items[i + 1:]:
                    counts[(left, right)] = counts.get((left, right), 0) + 1
        rows.extend({"Method": method, "ProteinA": pair[0], "ProteinB": pair[1], "SharedPathways": count}
                    for pair, count in counts.items())
    return pd.DataFrame(rows)


def covariate_availability():
    """Document which confounder checks the released matrices support."""
    return pd.DataFrame([
        {
            "Cohort": "Stanford PE (SomaLogic)", "Covariate": "Gestational stage",
            "Available": True, "AnalysisUse": "Quadratic time residualisation and grouped CV",
        },
        {
            "Cohort": "Stanford PE (SomaLogic)", "Covariate": "Maternal age/BMI/parity/smoking",
            "Available": False, "AnalysisUse": "Not present in the supplied processed matrix",
        },
        {
            "Cohort": "External LOPE (SomaLogic)", "Covariate": "Gestational age",
            "Available": True, "AnalysisUse": "Locked early-pregnancy inclusion (<=22 weeks)",
        },
        {
            "Cohort": "External LOPE (SomaLogic)", "Covariate": "Maternal age/BMI/parity/smoking",
            "Available": False, "AnalysisUse": "Not present in the public proteomic supplement",
        },
    ])


def main():
    if not SELECTIONS.exists():
        raise FileNotFoundError("Run src/run_expanded_benchmark.py before biological validation.")
    selections = pd.read_csv(SELECTIONS)
    selections = selections[selections.Dataset == "PE"].copy()
    frequency, panels = stable_panels(selections)
    pe_columns = pd.read_csv(PE, nrows=0).columns
    universe = {canonical_symbol(column) for column in pe_columns if column not in {"ID", "time", "label"}}
    known = known_marker_enrichment(panels, universe)
    reactome = reactome_enrichment(panels, universe)
    association = time_adjusted_associations(set().union(*panels.values()))
    edges = pathway_edges(reactome)
    frequency.to_csv(OUT / "selection_frequency.csv", index=False)
    known.to_csv(OUT / "known_pe_marker_enrichment.csv", index=False)
    reactome.to_csv(OUT / "reactome_enrichment.csv", index=False)
    association.to_csv(OUT / "time_adjusted_feature_associations.csv", index=False)
    edges.to_csv(OUT / "reactome_comembership_network.csv", index=False)
    covariate_availability().to_csv(OUT / "covariate_availability.csv", index=False)
    print(known.to_string(index=False))
    print(reactome.groupby("Method").head(3).to_string(index=False) if not reactome.empty else "No Reactome enrichment")


if __name__ == "__main__":
    main()
