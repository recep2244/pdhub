#!/usr/bin/env nextflow
/*
 * Protein Design Hub — reproducible predict → evaluate pipeline.
 *
 * Fans a multi-record FASTA out into one prediction per sequence (parallel),
 * evaluates each predicted structure, and collects all metrics into one JSON.
 * Each step shells out to the project CLI (`pdhub`), so the pipeline is the same
 * code path as the app — just orchestrated reproducibly by Nextflow.
 *
 * Usage:
 *   nextflow run pipelines/predict_evaluate.nf \
 *       --input sequences.fasta \
 *       --predictor esmfold \
 *       --outdir results/
 */

nextflow.enable.dsl = 2

params.input     = null            // multi-record FASTA
params.predictor = 'esmfold'       // esmfold | colabfold | chai1 | boltz2 | esm3
params.metrics   = 'clash_score,sasa,contact_energy,disorder'
params.outdir    = 'results'
params.pdhub     = 'pdhub'         // CLI entrypoint (override if not on PATH)

process SPLIT_FASTA {
    tag 'split'
    input:
    path fasta
    output:
    path 'seq_*.fasta'
    script:
    """
    awk '/^>/{n++} {print > sprintf("seq_%04d.fasta", n)}' ${fasta}
    """
}

process PREDICT {
    tag { fasta.baseName }
    publishDir "${params.outdir}/predictions", mode: 'copy'
    input:
    path fasta
    output:
    tuple val(fasta.baseName), path("${fasta.baseName}_pred/**/*.{pdb,cif}")
    script:
    """
    ${params.pdhub} predict single ${fasta} ${params.predictor} \
        --output ${fasta.baseName}_pred
    """
}

process EVALUATE {
    tag { id }
    publishDir "${params.outdir}/evaluation", mode: 'copy'
    input:
    tuple val(id), path(structure)
    output:
    path "${id}_metrics.json"
    script:
    """
    ${params.pdhub} evaluate run ${structure} \
        --metrics ${params.metrics} \
        --output ${id}_metrics.json
    """
}

process COLLECT {
    tag 'collect'
    publishDir "${params.outdir}", mode: 'copy'
    input:
    path metrics_files
    output:
    path 'all_metrics.json'
    script:
    """
    python - <<'PY'
    import json, glob
    out = {}
    for f in glob.glob("*_metrics.json"):
        try:
            out[f.replace("_metrics.json","")] = json.load(open(f))
        except Exception as e:
            out[f] = {"error": str(e)}
    json.dump(out, open("all_metrics.json","w"), indent=2)
    print(f"Collected {len(out)} result(s)")
    PY
    """
}

workflow {
    if( !params.input ) error "Provide --input <multi-record FASTA>"
    fasta = Channel.fromPath(params.input, checkIfExists: true)
    SPLIT_FASTA(fasta)
        | flatten
        | PREDICT
        | EVALUATE
        | collect
        | COLLECT
}
