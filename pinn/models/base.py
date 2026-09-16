# -*- coding: utf-8 -*-
"""Basic functions for PiNN models"""
import threading
import tensorflow as tf
from pinn.utils import pi_named

_infer_dtype = threading.local()


def tf_dtype_from_name(name='float32'):
    """Map TF dtype name ``float32`` / ``float64`` to a ``tf.DType``.

    ``None`` defaults to ``tf.float32``.
    """
    if name is None:
        name = 'float32'
    try:
        dtype = tf.as_dtype(name)
    except TypeError as exc:
        raise ValueError(
            f'Unknown dtype {name!r}. Expected float32 or float64.') from exc
    if dtype not in (tf.float32, tf.float64):
        raise ValueError(
            f'Unknown dtype {name!r}. Expected float32 or float64.')
    return dtype


def dtype_from_params(params):
    """Read ``settings.dtype``; missing means ``float32``."""
    settings = (params or {}).get('settings') or {}
    return settings.get('dtype', 'float32')


def apply_dtype(name='float32'):
    """Set Keras floatx and dtype policy (all new float weights/ops)."""
    dtype = tf_dtype_from_name(name)
    tf.keras.backend.set_floatx(dtype.name)
    tf.keras.mixed_precision.set_global_policy(dtype.name)
    return dtype.name


def set_infer_dtype(name):
    """Calculator sets this before ``predict()`` (MACE ``default_dtype``)."""
    _infer_dtype.name = name


class _CastSaver(tf.compat.v1.train.Saver):
    """Restore a checkpoint into variables of a (possibly different) dtype.

    TensorFlow variables cannot change dtype in place, so this is the
    equivalent of PyTorch ``model.float()`` / ``model.double()``: build
    the graph at the target dtype, then ``tf.cast`` each weight once at
    load. After that every op runs in the target dtype.
    """

    def __init__(self):
        super().__init__(var_list=[], allow_empty=True)

    def restore(self, sess, save_path):
        import numpy as np
        reader = tf.compat.v1.train.load_checkpoint(save_path)
        keys = reader.get_variable_to_shape_map()
        for var in tf.compat.v1.global_variables():
            key = var.op.name
            if key not in keys:
                continue
            raw = np.asarray(reader.get_tensor(key))
            dst = var.dtype.base_dtype.as_numpy_dtype
            # The Estimator finalizes the graph before the scaffold saver
            # restores, so no new ops may be created here: var.assign(ndarray)
            # would add a const + assign pair and raise "Graph is finalized".
            # load() feeds the value into the variable's existing initializer.
            var.load(raw.astype(dst, copy=False), sess)


def export_model(model_fn):
    # default parameters for all models
    from pinn.optimizers import default_adam
    default_settings = {'dtype': 'float32'}
    default_params = {'optimizer': default_adam, 'settings': default_settings}
    def pinn_model(params, **kwargs):
        model_dir = params['model_dir']
        params_tmp = default_params.copy()
        params_tmp.update(params)
        settings = dict(default_settings)
        settings.update(params.get('settings') or {})
        params_tmp['settings'] = settings
        params = params_tmp
        apply_dtype(dtype_from_params(params))
        def model_fn_with_dtype(features, labels, mode, params):
            dt = dtype_from_params(params)
            if mode == tf.estimator.ModeKeys.PREDICT:
                infer_dt = getattr(_infer_dtype, 'name', None) or dt
                apply_dtype(infer_dt)
                spec = model_fn(features, labels, mode, params)
                if tf_dtype_from_name(infer_dt) != tf_dtype_from_name(dt):
                    spec = spec._replace(
                        scaffold=tf.compat.v1.train.Scaffold(saver=_CastSaver()))
                return spec
            apply_dtype(dt)
            return model_fn(features, labels, mode, params)
        model = tf.estimator.Estimator(
            model_fn=model_fn_with_dtype, params=params,
            model_dir=model_dir, **kwargs)
        return model
    return pinn_model

class MetricsCollector():
    def __init__(self, mode):
        self.mode = mode
        self.LOSS = []
        self.ERROR = []
        self.METRICS = {}

    def add_error(self, tag, data, pred, mask=None, weight=1.0,
                  use_error=True, log_error=True, log_hist=True):
        """Add the error

        Args:
            tag (str): name of the error.
            data (tensor): data label tensor.
            pred (tensor): prediction tensor.
            mask (tensor): default to None (no mask, not implemented yet).
            weight (tensor): default to 1.0.
            mode: ModeKeys.TRAIN or ModeKeys.EVAL.
            return_error (bool): return error vector (for usage with Kalman Filter).
            log_loss (bool): log the error and loss function.
            log_hist (bool): add data and predicition histogram to log.
            log_mae (bool): add the mean absolute error to log.
            log_rmse (bool): add the root mean squared error to log.
        """
        error = data - pred
        weight = tf.cast(weight, data.dtype)
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            if log_hist:
                tf.compat.v1.summary.histogram(f'{tag}_DATA', data)
                tf.compat.v1.summary.histogram(f'{tag}_PRED', pred)
                tf.compat.v1.summary.histogram(f'{tag}_ERROR', error)
            if log_error:
                mae = tf.reduce_mean(tf.abs(error))
                rmse = tf.sqrt(tf.reduce_mean(error**2))
                tf.compat.v1.summary.scalar(f'{tag}_MAE', mae)
                tf.compat.v1.summary.scalar(f'{tag}_RMSE', rmse)
            if mask is not None:
                error, weight = _apply_mask(error, weight, mask)
            if use_error:
                loss = tf.reduce_mean(error**2 * weight)
                tf.compat.v1.summary.scalar(f'{tag}_LOSS', loss)
                self.ERROR.append(error*tf.math.sqrt(weight))
                self.LOSS.append(loss)
        if self.mode == tf.estimator.ModeKeys.EVAL:
            if log_error:
                self.METRICS[f'METRICS/{tag}_MAE'] = tf.compat.v1.metrics.mean_absolute_error(data, pred)
                self.METRICS[f'METRICS/{tag}_RMSE'] = tf.compat.v1.metrics.root_mean_squared_error(data, pred)
            if mask is not None:
                error, weight = _apply_mask(error, weight, mask)
            if use_error:
                loss = tf.reduce_mean(error**2 * weight)
                self.METRICS[f'METRICS/{tag}_LOSS'] = tf.compat.v1.metrics.mean(loss)
                self.LOSS.append(loss)


def _apply_mask(error, weight, mask):
    """Select the kept entries of an error and of a per-entry weight.

    ``tf.boolean_mask`` flattens what it selects, so a force error of shape
    ``[n_atoms, 3]`` comes back as ``[n_kept]``. A weight of the same rank --
    ``f_weights`` under ``use_f_weights`` -- has to travel through the same
    selection, or the ``error**2 * weight`` that follows tries to broadcast
    ``[n_kept]`` against ``[n_atoms, 3]`` and the graph fails to build. Scalar
    weights (a plain loss multiplier) broadcast fine and are passed through.

    Returns:
        tuple: the masked error and the weight to multiply it by.
    """
    if tf.is_tensor(weight) and weight.shape.rank == error.shape.rank:
        weight = tf.boolean_mask(weight, mask)
    return tf.boolean_mask(error, mask), weight


@pi_named('TRAIN_OP')
def get_train_op(optimizer, metrics, tvars, separate_errors=False):
    """
    Args:
        optimizer: a PiNN optimizer config.
        params: optimizer parameters.
        loss: scalar loss function.
        error: a list of error vectors (reserved for EKF).
        network: a PiNN network instance.
        sperate_errors (bool): separately update elements in the metrics
    """
    from pinn.optimizers import get, EKF, gEKF
    import numpy as np

    optimizer = get(optimizer)
    optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()
    nvars = np.sum([np.prod(var.shape) for var in tvars])
    print(f'{nvars} trainable vaiables, training with {tvars[0].dtype.name} precision.')

    if not (isinstance(optimizer, EKF) or isinstance(optimizer, gEKF)):
        loss_list =  metrics.LOSS
        if separate_errors:
            selection = tf.random.uniform([], maxval= len(loss_list), dtype=tf.int32)
            loss = tf.stack(loss_list)[selection]
        else:
            loss = tf.reduce_sum(loss_list)
        grads = tf.gradients(loss, tvars)
        return optimizer.apply_gradients(zip(grads, tvars))
    else:
        error_list =  metrics.ERROR
        # EKF error vectors are scaled
        if isinstance(optimizer, EKF):
            error = tf.concat([tf.reshape(e, [-1])/tf.math.sqrt(tf.cast(tf.size(e), e.dtype))
                               for e in error_list], 0)
        # gEKF should handle this automatically
        if isinstance(optimizer, gEKF):
            error = tf.concat([tf.reshape(e, [-1]) for e in error_list], 0)
        if separate_errors:
            selection = tf.random.uniform([], maxval= len(error_list), dtype=tf.int32)
            mask = tf.concat([tf.fill([tf.size(e)], tf.equal(selection,i))
                              for i,e in enumerate(error_list)], 0)
            error = tf.boolean_mask(error, mask)
        return optimizer.get_train_op(error, tvars)
