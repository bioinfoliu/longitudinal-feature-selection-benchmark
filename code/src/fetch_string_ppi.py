"""Extract a STRING physical-interaction subnetwork entirely offline.

The candidate panel never leaves the workspace. The script queries the public
human STRING v12 physical links downloaded under ``data/external/raw``.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd

from protein_symbols import canonical_symbol


ROOT = Path(__file__).resolve().parents[1]
FREQUENCY = ROOT / "results" / "biological_validation" / "selection_frequency.csv"
INFO = ROOT / "data" / "external" / "raw" / "9606.protein.info.v12.0.txt.gz"
LINKS = ROOT / "data" / "external" / "raw" / "9606.protein.physical.links.v12.0.txt.gz"
OUT = ROOT / "results" / "biological_validation" / "string_physical_ppi.tsv"
MAPPING_OUT = ROOT / "results" / "biological_validation" / "string_panel_mapping.csv"


def stable_snn_panel(n: int = 20) -> list[str]:
    frequency = pd.read_csv(FREQUENCY)
    subset = frequency[frequency.Method == "SNN"].copy()
    subset["Gene"] = subset.Feature.map(canonical_symbol)
    subset = subset[subset.Gene != ""]
    return list(dict.fromkeys(subset.sort_values(
        ["SelectionFrequency", "Feature"], ascending=[False, True]
    ).Gene))[:n]


def main(minimum_score: int = 400) -> None:
    for path in (FREQUENCY, INFO, LINKS):
        if not path.exists():
            raise FileNotFoundError(path)

    genes = stable_snn_panel()
    info = pd.read_csv(INFO, sep="\t", compression="gzip")
    id_column = "#string_protein_id"
    info["CanonicalGene"] = info.preferred_name.map(canonical_symbol)
    mapping = info[info.CanonicalGene.isin(genes)].drop_duplicates("CanonicalGene")
    gene_to_id = dict(zip(mapping.CanonicalGene, mapping[id_column]))
    id_to_gene = {value: key for key, value in gene_to_id.items()}

    mapping_table = pd.DataFrame({
        "Gene": genes,
        "STRINGProteinID": [gene_to_id.get(gene, "") for gene in genes],
        "Mapped": [gene in gene_to_id for gene in genes],
    })
    MAPPING_OUT.parent.mkdir(parents=True, exist_ok=True)
    mapping_table.to_csv(MAPPING_OUT, index=False)

    ids = set(id_to_gene)
    chunks = []
    with gzip.open(LINKS, "rt", encoding="utf-8") as handle:
        for chunk in pd.read_csv(handle, sep=r"\s+", chunksize=500_000):
            keep = (
                chunk.protein1.isin(ids)
                & chunk.protein2.isin(ids)
                & (chunk.combined_score >= minimum_score)
            )
            if keep.any():
                chunks.append(chunk.loc[keep].copy())

    if chunks:
        edges = pd.concat(chunks, ignore_index=True)
        edges["preferredName_A"] = edges.protein1.map(id_to_gene)
        edges["preferredName_B"] = edges.protein2.map(id_to_gene)
        edges["score"] = edges.combined_score / 1000.0
        edges["Pair"] = edges.apply(
            lambda row: "|".join(sorted([row.preferredName_A, row.preferredName_B])), axis=1
        )
        edges = edges.sort_values("score", ascending=False).drop_duplicates("Pair")
        edges = edges[["preferredName_A", "preferredName_B", "score", "protein1", "protein2"]]
        edges = edges.sort_values("score", ascending=False)
    else:
        edges = pd.DataFrame(columns=[
            "preferredName_A", "preferredName_B", "score", "protein1", "protein2"
        ])
    edges.to_csv(OUT, sep="\t", index=False)

    print("SNN panel:", ";".join(genes))
    print(f"STRING mapped: {mapping_table.Mapped.sum()}/{len(mapping_table)}")
    print(f"Physical edges (score >= {minimum_score}): {len(edges)}")
    if not edges.empty:
        print(edges[["preferredName_A", "preferredName_B", "score"]].to_string(index=False))


if __name__ == "__main__":
    main()
