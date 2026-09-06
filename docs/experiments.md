# Experiments and limits

These are records of exploratory development, not validated scientific benchmark claims. Source curation remained provisional; no sealed test was evaluated.

## Encoder transfer

The matched direct experiment used 2,404 rat-plasma training points from 11 compounds and 1,188 validation points from four compounds. Newly reserved chemistry removed 2,048 formerly training-assigned points from both arms. Both used identical preprocessing, optimizer, minibatch order per seed, decoder initialization and development checkpoint selection. Only encoder initialization differed.

Across three seeds the direct model scored 26.27% versus 24.32% from scratch. The hybrid scored 14.54% versus 14.12%. One seed worsened in each experiment. All transferred runs used one fixed auxiliary checkpoint, so the three seeds are not independent pretraining replicates. These small changes do not establish superiority, and different historical cohorts must not be compared as if identical.

## Auxiliary properties

The expanded experiment used 5,681 training labels from 2,571 compounds and 1,422 auxiliary-validation labels from 632 compounds across 18 tasks. The equal-task within-twofold average was 42.60% for the neural encoder, 42.30% for ExtraTrees and 38.40% for a per-task training-median baseline. Some tasks have very small validation counts.

PPBR targets were expressed as log unbound fraction, not log binding percentage. Clearance, distribution volume, protein binding and assay-specific units were kept distinct. Conflict/duplicate filtering and chemical reservations preceded training. Checkpoint selection and reporting used the same development validation; this is not external-test performance.

## Reproducibility boundary

The repository includes aggregate CSV summaries, reusable model and evaluation code, an architecture-comparison workflow, numerical tests and a runnable synthetic example. It does not contain the original datasets, chemistry caches, checkpoints, experiment-specific local paths or all acquisition scripts. Historical experiments cannot be rerun from the aggregates alone. Do not interpret this as a packaged reproduction of PKSmart, AstraZeneca, Novartis or any other published model.

## Directions worth investigating

Prioritize independently reviewed concentration data, explicit use cases and matched experiments over larger models. Study subject/compound uncertainty, separate observed organs from latent compartments, and report negative results. Prospective validation and an adequately sized untouched test set would be needed before claims of utility for unseen human chemistry.
