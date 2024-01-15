#!/usr/bin/env nextflow

params.train_flags = '--init --shuffle-buffer 1000 --train-steps 3000000 -b 30'
params.convert_flags = '-o train:8,eval:2'

process train {
  publishDir "models/$name", pattern: "{model,*.log}"
  label "train"
  tag "$name"

  input:
  tuple(path(ds), path(input), val(seed))

  output:
  path ("$name/", type:'dir')
  path ("{train,eval}.log")
  path ("{train,eval}.{yml,tfr}")

  script:
  name = "${input.baseName}-${ds[0].baseName}-${seed}"
  """
  pinn convert ${ds[0].baseName}.yml ${params.convert_flags} --seed $seed
  pinn train $input -d $name ${params.train_flags}
  pinn log $name/eval --tag '' > eval.log
  pinn log $name --tag '' > train.log
  """
}

