# paper

Experiments on estimating piano-score difficulty (grades 1–9) on a shared set of 806 scores. Each folder is a
separate project with its own `README.md` explaining how to run it and what the folder is about.

## Data (repo root)

- `score_musicxml/`: the 806 scores, `score_<id>.musicxml` (not tracked in git).
- `806_2_score_dataset_clean.xlsx`: one row per score with the difficulty label, composer, title and tempo info.
  The Docker projects mount it read-only under this name. `806_2_score_dataset_sample.xlsx` is a small sample.

## Projects

| Folder | What it does | Runs with |
|---|---|---|
| [`cipi_model/`](cipi_model/README.md) | Pretrained CIPI models (Ramoneda et al., ESWA 2023) | Docker |
| [`rubricnet_model/`](rubricnet_model/README.md) | Pretrained RubricNet (Ramoneda et al., ISMIR 2024) | Docker |
| [`nina_model/`](nina_model/README.md) | Inference with our ONSET difficulty model | Docker |
| [`model_comparison/`](model_comparison/README.md) | Where CIPI and RubricNet fail on the same scores | Python |
| [`nina_stats/`](nina_stats/README.md) | Dataset statistics and hand/coordination features for the paper | Python |

Docker projects: `cd <folder> && docker compose build && docker compose run --rm <service>`.
Results are written to each project's `results/` folder.
