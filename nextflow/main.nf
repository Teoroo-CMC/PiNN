#!/usr/bin/env nextflow

params.seeds = 3

include {gen_qm9; gen_md17; gen_rmd17} from './datasets.nf'
include {train} from './train.nf'

dataset_loader = [
  'qm9': gen_qm9,
  'rmd17': gen_rmd17
]

workflow {
  ch_seeds = Channel.of(1..params.seeds)
  ch_batch_size = Channel.of(64)
  todo_channel = Channel.of()
  params.setup.each { entry -> 
    dataset = entry.key
    println "$dataset"
    pot = entry.value['pot']
    dataset_loader[dataset]
    | combine(Channel.fromPath("./inputs/$pot"))
    | combine(ch_seeds)
    | combine(ch_batch_size)
    | map {[[it[0],it[1]],it[2], it[3], it[4]]}
    | set {task4dataset}
    todo_channel = todo_channel.mix(task4dataset)
  }

  todo_channel.view()
  
}

