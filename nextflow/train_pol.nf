#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// Adjustables
params.inputs  = '../inputs/eem.yml'      // inputs to use, can contain wildcard
params.dataset = './coords/qm7b.xyz'   // training set to use, can contain wildcard
params.steps   = '500000'              // max number of steps, comma separated
params.seeds   = '1'                   // random seed for spliting, comma separated

//
model_config = Channel.fromPath(params.inputs)
    .combine(Channel.fromPath(params.dataset))
    .combine(Channel.fromList(params.steps.tokenize(',')))
    .combine(Channel.fromList(params.seeds.tokenize(',')))


workflow {
    get_qm7b()
    pinn_train(model_config)
}

def shorten(x) {sprintf('%.5E', (double) x.toInteger()).replaceAll(/\.?0*E/, 'E').replaceAll(/E\+0*/, 'E')}

process get_qm7b {
  publishDir "coords/"

  output:
  path("qm7b.xyz")

  script:
  """
  #!/usr/bin/env python

  import requests

  url = "https://archive.materialscloud.org/record/file?record_id=84&filename=qm7b_coords.xyz"
  r = requests.get(url)

  with open('coords/qm7b.xyz', "wb") as f:
    f.write(r.content)
  """
}

process pinn_train {
    publishDir 'pol_models/'
    tag "$ds.baseName-$input.baseName-${shorten(step)}-$seed"

    input:
    tuple (file(input), file(ds), val(step), val(seed))

    output:
    path "$ds.baseName-$input.baseName-${shorten(step)}-$seed", type:'dir'

    """
    #!/usr/bin/env python
    import yaml
    from pinn.io import load_qm7b
    from pinn.models.pol_models import *
    from pinn.io import sparse_batch

    def pre_fn(tensors):
        with tf.name_scope("PRE") as scope:
            network = get_network(params['network'])
            tensors = network.preprocess(tensors)
        return tensors

    dataset = lambda: load_qm7b('qm7b.xyz', splits={'train':8, 'test':2}, seed=$seed)
    train = lambda: dataset()['train'].apply(sparse_batch(30)).map(pre_fn).cache().repeat().shuffle(1000)
    test = lambda: dataset()['test'].apply(sparse_batch(30)).map(pre_fn)

    config = tf.estimator.RunConfig(keep_checkpoint_max=1,
                                    log_step_count_steps=1000,
                                    save_summary_steps=1000,
                                    save_checkpoints_steps=10000)

    train_spec = tf.estimator.TrainSpec(input_fn=train, max_steps=$step)
    eval_spec = tf.estimator.EvalSpec(input_fn=test, throttle_secs=600)
    with open('${input.baseName}.yml') as f:
        params = yaml.safe_load(f)
        params['model_dir'] = '$ds.baseName-$input.baseName-${shorten(step)}-$seed'
    model = pinn.get_model(params, config=config)
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    """
}
