# Command-line interface

PiNN provides a command-line tool: `pinn_train` to train models. The program was
meant to work with the Google Cloud AI platform, but it should also run well on
local machines.

The trainer effectively runs train_and_evaluate with the given dataset. To use
the trainer, both training data and evaluation data must be prepared as a
tfrecord format. Currently, only training potentials is supported. To see the
available options, run `pinn_train --help`.

Example usage of ``pinn_train`` on a local machine:

```bash
pinn_trian --model-dir=my_model --params=params.yml \\
           --train-data=train.yml --eval-data=test.yml \\
           --train-steps=1e6 --eval-steps=100 \\
           --cache-data=True
```


Example usage of ``pinn_train`` on Google Cloud: 

It is recommanded to use our docker image. Since we do not serve it on
Google Container Registery, you'll need to build one yourself (suppose
you have an active Gclound project)::

```bash
gcloud auth configure-docker
docker pull yqshao/pinn:dev
docker tag yqshao/pinn:dev gcr.io/my-proj/pinn:cpu
docker push gcr.io/my-proj/pinn:cpu 
```

To submit a job on GCloud::

```bash
gcloud ai-platform jobs submit training my_job_0 \\
       --region europe-west1 --master-image-uri gcr.io/my-proj/pinn:cpu \\
       -- \\
       --model-dir=gs://my-bucket/models/my_model_0 \\
       --params-file=gs://my-bucket/models/params.yml \\
       --train-data=gs://my-bucket/data/train.yml \\
       --eval-data=gs://my-bucket/data/test.yml \\
       --train-steps=1000 --cache-data=True
```


