#!/usr/bin/env nextflow

params.train_flags = "--init --shuffle-buffer 1000 --train-steps 2000000"
params.convert_flags = '-o train:950,eval:50'

process train {
  publishDir "models/$name"
  label "train"
  tag "$name"

  input:
  tuple(path(ds), path(input), val(seed), val(batch_size))

  output:
  path ("$name/", type:'dir')
  path ("{train,eval}.log")

  script:
  name = "${input.baseName}-${ds[0].baseName}-bs${batch_size}-${seed}"
  """
  pinn convert ${ds[0].baseName}.yml ${params.convert_flags} --total 1000 --seed $seed
  pinn train $input -d $name ${params.train_flags} -b ${batch_size}
  pinn log $name/eval --tag '' > eval.log
  pinn log $name --tag '' > train.log
  """
}

