#!/usr/bin/env nextflow

params.seeds = 3

include {gen_qm9; gen_md17; gen_rmd17} from './datasets.nf'
include {train} from './train.nf'

dataset_loader = [
  'qm9': gen_qm9,
  'rmd17': gen_rmd17
]

// pots = [
//   'pinet_pot_nofrc': Channel.fromPath("./inputs/*-nofrc.yml"),
//   'pinet_pot': Channel.fromPath("./inputs/pinet-pot.yml"),
//   'pinet2_pot': Channel.fromPath("./inputs/pinet2-pot-{simple,general}.yml")
//   'pinet2_pot_frc': Channel.fromPath("./inputs/pinet2-pot-{simple,general}-frc.yml")
// ]

workflow {
  ch_seeds = Channel.of(1..params.seeds)
  ch_batch_size = Channel.of(64)
  // ch_pinet_pot_nofrc = Channel.fromPath("./inputs/*-nofrc.yml")
  // ch_pinet_pot = Channel.fromPath("./inputs/pinet-pot.yml")
  // ch_pinet2_pot = Channel.fromPath("./inputs/pinet2-pot-{simple,general}.yml")
  // ch_pinet2_pot_frc = Channel.fromPath("./inputs/pinet2-pot-{simple,general}-frc.yml")

  params.setup.each { entry -> 
    dataset = entry.key
    println "$dataset"
    pot = entry.value['pot']
    dataset_loader[dataset]
    | combine(Channel.fromPath("./inputs/$pot"))
    | combine(ch_seeds)
    | combine(ch_batch_size)
    | set {dataset}
    dataset
    | map {[[it[0],it[1]],it[2], it[3], it[4]]}
    | view
    // | train
  }
  
  // gen_qm9()
  // gen_md17()

  // gen_qm9.out
  // | combine(ch_pinet_pot_nofrc)
  // | combine(ch_seeds)
  // | combine(ch_batch_size)
  // | set {ch_qm9}

  // gen_md17.out
  // | combine(ch_pinet2_pot_frc.concat(ch_pinet2_pot).concat(ch_pinet_pot).concat(ch_pinet_pot_nofrc))
  // | combine(ch_seeds)
  // | combine(ch_batch_size)
  // | set {ch_md17}
  // ch_md17 //.concat(ch_rmd17)
  // | map {[[it[0],it[1]],it[2], it[3], it[4]]}
  // | train
}
