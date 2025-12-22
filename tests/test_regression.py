# -*- coding: utf-8 -*-

"""This module tests that trained models and checkes that the
evaluation matches what is saved in the evaluation log.

Versioned regression test data is stored at:
https://www.dropbox.com/scl/fo/tkq0zumunvofp7dckfta5/AC_WcVOW-ihchdKTTVOHrno?rlkey=6nz86x7eflli58rlajfspzf8v&dl=0 

Upon a breaking change, a new set of reference model and dataset
should be generated, and the url for the test should be updated.

"""

import io, os, pytest, requests, tarfile
from glob import glob

compat_version = 'v1.1'
compat_url = 'https://www.dropbox.com/scl/fi/isevisxxc4ts3nf7v0guy/v1.1_datasets.tar?rlkey=7sigw34vcjuia0vt45g13gzml&dl=1'
cache_path = os.path.expanduser(f'~/.cache/pinn/regression/{compat_version}')

if not os.path.exists(cache_path):
    print(f"Downloading regression test to cache dir: {cache_path}")
    cache_bytes = requests.get(compat_url, allow_redirects=True).content
    cache_fobj = io.BytesIO(cache_bytes)
    cache_fobj.seek(0)
    file = tarfile.open(fileobj=cache_fobj, mode='r:')
    os.makedirs(cache_path)
    file.extractall(path=cache_path)

model_list = glob(f'{cache_path}/*/')

@pytest.mark.parametrize('model_path', model_list)
@pytest.mark.forked
def test_model_regression(model_path):
    import pinn
    import numpy as np
    from pinn.io import load_ds,sparse_batch
    from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader

    # Run evaluation loop
    model = pinn.get_model(f'{model_path}/model')
    eval_fn = lambda: load_ds(f'{model_path}/eval.yml').apply(sparse_batch(30))
    eval_scores = model.evaluate(eval_fn)

    # Get reference scores
    ref_scores = {}
    logs = sorted(glob(f'{model_path}/model/eval/events.out.*'), key=os.path.getmtime)
    for log in logs[1:]: # clean up evaluation history
        os.remove(log)
    events = LegacyEventFileLoader(logs[0]).Load()
    for event in events:
        for v in event.summary.value:
            if ('RMSE' not in v.tag) and ('MAE' not in v.tag):
                continue
            if event.step==eval_scores['global_step']:
                ref_scores[v.tag] = v.simple_value
    print(eval_scores, '\n', ref_scores)
    for k in ref_scores.keys():
        assert np.allclose(ref_scores[k], eval_scores[k], rtol=0.01)
