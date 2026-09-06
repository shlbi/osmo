# Data contracts

The self-contained demo needs no external data. Research scripts require data you obtain and review separately. This release includes code and aggregate metrics, not a licensed training corpus.

## Core prepared concentration table

The existing baseline/architecture workflows read `processed/latest.json` with a `dataset_id`, then `processed/<dataset_id>/observations.parquet` and `compounds.parquet` under the supplied data root. The CvTdb preparation code defines the full schema.

Concentration observations include `compound_id`, `smiles`, `cluster_id`, `curve_id`, `species`, `tissue`, `time_h`, `concentration`, `dose_mg_kg`, `route`, `infusion_h`, `formulation` and `split`. Some diagnostics also require `study_id` and `sex`. Core historical rat-plasma training expects positive concentrations in ng/mL and doses in mg/kg. Do not insert ng/g brain data into that contract.

For schema discovery, inspect `src/osmo/cvtdb.py`. The `osmo acquire` command downloads a public source; acquisition is not a statement that every downloaded observation is training-ready or redistributable. Review the source's terms and the preparation assumptions before using it.

## Example research commands

After installation and source review:

```bash
osmo --data-root data --reports reports acquire
osmo --data-root data --reports reports prepare
osmo --data-root data --reports reports train-baseline --species rat --tissue plasma
python scripts/compare_architectures.py --root data --output reports/architectures --arms direct cmt2
```

The pipeline can contact external APIs to resolve chemical structures. Public services and source files may change; preserve hashes and versions for your own run. The historical aggregate scores are not guaranteed to reproduce from a fresh download.

## Expanded adapters

`expanded.py` supports reviewed long-form records carrying explicit `unit`, `dose_unit`, `route`, `species`, `matrix`, `compound_id`, `arm`, `value`, `dose`, `time_h` and eligibility flags. It emits separate output-head labels for species, matrix and unit. This is a data adapter, not a general human-organ model.

Keep censored observations separate from numeric point targets. Preserve reported zeros without arbitrary log transforms. Never treat fitted curves, generated animal properties or simulations as independent observations. Paired periods and related metabolites need appropriate study and chemistry grouping.

## Legacy PK-Sim files

The narrow reader in `osp_binary.py` handles observed numeric arrays in supported local project formats. It does not load an executable .NET object graph. It does decompress data and parse XML, so it should be used only with trusted files and suitable resource limits. PK-Sim software and observed data have separate upstream terms.
