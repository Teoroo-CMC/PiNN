#!/usr/bin/env nextflow

params.seeds = 3

include {gen_qm9; gen_md17} from './datasets.nf'
include {train} from './train.nf'

workflow {
  ch_seeds = Channel.of(1..params.seeds)
  ch_pinet_pot_nofrc = Channel.fromPath("./inputs/pinet-pot-nofrc.yml")
  ch_pinet_pot = Channel.fromPath("./inputs/pinet-pot.yml")
  
  gen_qm9()
  gen_md17()

  gen_qm9.out
  | combine(ch_pinet_pot_nofrc)
  | combine(ch_seeds)
  | set {ch_qm9}

  gen_md17.out
  | combine(ch_pinet_pot)
  | combine(ch_seeds)
  | set {ch_md17}

  ch_qm9.concat(ch_md17)
  | map {[[it[0],it[1]],it[2],it[3]]}
  | train
}
