# Data Provenance

This directory contains fixture data used by the ProteinEnv test suite and task loaders.

## Fixtures

### `easy_proteins.json`

| Field | Description |
|-------|-------------|
| `protein_id` | UniProt accession |
| `name` | Official UniProt protein name |
| `sequence` | Full canonical amino-acid sequence (single-letter codes) |
| `family` | Correct UniProt protein family classification |
| `family_choices` | 10 candidate families (1 correct + 9 plausible distractors) |
| `source` | Data provenance string |
| `difficulty` | Always `"easy"` |

**Source:** UniProt Swiss-Prot, reviewed entries. Sequences retrieved from
canonical FASTA records. Family labels sourced from UniProt "Family & Domains"
annotations. Distractor families are real UniProt protein families selected
for plausibility within the same broad functional category.

**Proteins included:**
P01308 (Insulin), P68871 (Haemoglobin beta), P00533 (EGFR), P04637 (p53),
P60709 (Beta-actin), P00441 (SOD1-CuZn), P01023 (Alpha-2-macroglobulin),
P02768 (Serum albumin), P00734 (Prothrombin), P35222 (Beta-catenin).

---

### `medium_proteins.json`

| Field | Description |
|-------|-------------|
| `protein_id` | UniProt accession |
| `name` | Official UniProt protein name |
| `sequence` | Canonical amino-acid sequence |
| `go_terms` | Object keyed by GO namespace with lists of GO term IDs |
| `source` | Data provenance string |
| `difficulty` | Always `"medium"` |

**Source:** UniProt Swiss-Prot reviewed entries cross-referenced with
QuickGO annotations (EMBL-EBI). GO term IDs follow the standard `GO:XXXXXXX`
format (7-digit zero-padded identifier). Only experimentally validated or
computationally inferred HIGH-CONFIDENCE terms are included.

**Proteins included:**
P04637 (p53), P00533 (EGFR), P68871 (HBB), P07900 (HSP90-alpha),
P38398 (BRCA1), P15056 (BRAF), P00441 (SOD1), P60709 (ACTB),
P02768 (ALB), P01308 (INS).

---

### `hard_variants.json`

| Field | Description |
|-------|-------------|
| `variant_id` | dbSNP rsID or ClinVar variant accession |
| `gene` | HGNC gene symbol |
| `protein_id` | UniProt accession of the affected protein |
| `wildtype_aa` | Reference amino acid (single letter) |
| `mutant_aa` | Alternate amino acid (single letter) |
| `position` | 1-indexed position in the canonical isoform |
| `sequence_with_mutation` | Full protein sequence with the substitution applied |
| `pathogenicity` | ClinVar five-tier classification |
| `associated_diseases` | List of OMIM/MeSH disease names |
| `clinvar_id` | Numeric ClinVar variation ID |
| `source` | Data provenance string |
| `difficulty` | Always `"hard"` |

**Source:** ClinVar (NCBI), accessed 2024. Pathogenicity calls reflect the
clinical significance recorded in ClinVar at data-snapshot time. Variant
positions follow UniProt canonical isoform numbering.

**Variants included:**
TP53 R175H, BRCA1 C61G, EGFR T790M, BRAF V600E, KRAS G12D,
SOD1 A4V, PTEN R130Q, IDH1 R132H, PIK3CA H1047R, ABL1 T315I.

---

## Usage

Fixtures are loaded in `tests/conftest.py` and in `tasks/` loaders via
`json.load()`. They are **static** — do not modify them between runs, as
determinism is required for reproducible reward calculations.

## Licence

Sequence data derived from UniProt is provided under the
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
licence. GO term data from QuickGO (EMBL-EBI) is provided under
[CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).
ClinVar data is in the public domain (NCBI).
