#!/usr/bin/env nextflow

params.instances = 3
params.seeds = 100
params.batch_size = 100
params.public_dir = "models"

include {gen_qm9; gen_md17; gen_rmd17; gen_mpc; gen_perovskites_formation; gen_mpf; gen_lowsym; gen_mpc_db} from './datasets.nf'
include {train} from './train.nf'

dataset_loader = [
  'qm9': gen_qm9,
  'rmd17': gen_rmd17,
  'mp2018': gen_mp2018,
  'mp2021': gen_mp2021,
]

workflow {
  ch_seeds = Channel.of(1..params.seeds).randomSample(params.instances)
  ch_batch_size = Channel.from(params.batch_size)
  ch_eval_batch_size = Channel.of(params.eval_batch_size)
  todo_channel = Channel.of()
  params.setup.each { entry -> 
    dataset = entry.key
    println "$dataset"
    pot = entry.value['pot']
    dataset_loader[dataset]
    | combine(Channel.fromPath("./inputs/$pot"))
    | combine(ch_seeds)
    | combine(ch_batch_size)
    | map {[[it[0],it[1]],it[2], it[3], it[4], it[5]]}
    | set {task4dataset}
    todo_channel = todo_channel.mix(task4dataset)
  }

  todo_channel.view()
  | train
}

