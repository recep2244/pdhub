# Reproducible pipelines (Nextflow)

`predict_evaluate.nf` runs the **same CLI code path** as the app, but orchestrated
reproducibly: a multi-record FASTA is fanned out to one prediction per sequence
(parallel), each structure is evaluated, and all metrics are collected into a
single `all_metrics.json` — with timeline/trace/report provenance for free.

## Prerequisites
- [Nextflow](https://www.nextflow.io/) ≥ 23.10 (`curl -s https://get.nextflow.io | bash`)
- `pdhub` CLI on PATH (`pip install -e .` from the repo root), or pass `--pdhub /path/to/pdhub`

## Run
```bash
# local
nextflow run pipelines/predict_evaluate.nf \
    --input sequences.fasta \
    --predictor esmfold \
    --outdir results/

# GPU box (serialise prediction, parallelise evaluation)
nextflow run pipelines/predict_evaluate.nf -profile gpu \
    --input sequences.fasta --predictor chai1
```

## Parameters
| Param | Default | Notes |
|-------|---------|-------|
| `--input` | _(required)_ | multi-record FASTA |
| `--predictor` | `esmfold` | `esmfold`/`colabfold`/`chai1`/`boltz2`/`esm3` |
| `--metrics` | `clash_score,sasa,contact_energy,disorder` | comma-separated, see `pdhub evaluate metrics` |
| `--outdir` | `results` | predictions/, evaluation/, all_metrics.json, provenance reports |
| `--pdhub` | `pdhub` | CLI entrypoint override |

## Outputs
```
results/
├── predictions/            # one *_pred/ per sequence
├── evaluation/             # one *_metrics.json per structure
├── all_metrics.json        # merged metrics for all designs
├── pipeline_report.html    # resource usage
├── pipeline_timeline.html  # per-process timeline
└── pipeline_trace.txt      # machine-readable trace
```

Resume a partially-completed run with `-resume` (Nextflow caches completed tasks).
