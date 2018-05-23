from glob import glob
import tensorflow as tf


def test_ani_dataset():
    from pinn.dataset import ANI_H5_dataset
    files = glob('examples/ani/*.h5')
    dataset = ANI_H5_dataset(files)
    train = dataset.get_train(tf.float32)
    data = train.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        for i in range(10):
            tensors = sess.run(data)
    assert tensors['coord'].shape[-1] == 3


def test_pinn_model():
    from pinn.models import PiNN
    from pinn.dataset import ANI_H5_dataset

    files = glob('examples/ani/*.h5')
    dataset = ANI_H5_dataset(files)

    train_spec = tf.estimator.TrainSpec(
        input_fn=dataset.get_train, max_steps=10)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=dataset.get_vali)

    estimator = PiNN('tmp/PiNN-ANI')
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def test_pinn_calculator():
    from pinn.models import PiNN
    from pinn.calculator import PiNN_calc
    from ase.collections import g2
    from ase.optimize import BFGS

    model = PiNN('tmp/PiNN-ANI')
    calc = PiNN_calc(model=model)
    water = g2['H2O']
    water.set_calculator(calc)
    dyn = BFGS(water)
    dyn.run(fmax=0.05)
