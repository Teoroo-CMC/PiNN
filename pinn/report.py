#!/usr/bin/env python

"""
Script to generate reports from benchmar results
"""
import re, os
import warnings
import numpy as np
from typing import List
from glob import glob
from pathlib import Path
from collections import defaultdict

hartree2meV = 27.2114079527 * 1000
kcalpermol2meV = 0.0433641153087705 * 1000

def report_log(model_paths:List[str], keyword_filter:List[str]=[], log_name:str = 'eval.log'):
    """
    report log of training process

    Args:
        model_paths (List[str]): path the model host, e.g. models/pinet2-pot-simple-nofrc-qm9-bs120-1. The identifier of this model is [pinet2, pot, simple, nofrc, qm9, bs120], but exclude the last number.

        log_name (str, optional): name of log file. Defaults to 'eval.log'.

        keyword_filter (List[str], optional): filter option, only report the model contains all the args. Defaults to [].
    """
    # groupby name of model
    trials = defaultdict(list) # {name: [path1, path2, ...]}
    get_params = lambda x: tuple(x.split('-')[:-1])
    for path in model_paths.glob("eval.log"):
        path = Path(path)
        params = get_params(path.stem)

        if all([kw in params for kw in keyword_filter]):
            trials[params].append(path / log_name)

    results = {}
    print(f'\n# === status ===')
    print(f'#{" === model":<50} {"complete / total":>15}\n')
    for params, logs in trials.items():
        E_MAE = []
        F_MAE = []
        steps = []
        for log in logs:
            eval_log_arr = np.loadtxt(log)
            if eval_log_arr.ndim == 1:
                warnings.warn(f'{log} not start')
                continue
            E_MAE.append(eval_log_arr[-1, 2])
            F_MAE.append(eval_log_arr[-1, 7])
            steps.append(eval_log_arr[-1, 0])
        steps = np.array(steps)
        mask = steps == np.max(steps)
        complete_trial_ratio = f"{np.sum(mask)} / {len(mask)}"
        print(f'{"-".join(params):<50} : {complete_trial_ratio:>10}')
        results[params] = [np.mean(E_MAE), np.std(E_MAE), np.mean(F_MAE), np.std(F_MAE)]

    print(f'\n# === performance ===')
    print(f'# === {"model":<50} {"E_MAE(std)":>11} {"F_MAE(std)":>15}\n')
    for params, v in sorted(results.items()):
        key = ''.join(params)
        e_mae_and_std = f'{v[0]:.2f}({v[1]:.2f})'
        f_mae_and_std = f'{v[2]:.2f}({v[3]:.2f})'
        print(f'{key:<50}:  {e_mae_and_std:>15} {f_mae_and_std:>15}')


def report_qm9():
    logs = [np.loadtxt(logf) for logf in glob('models/pinet2-pot-simple-nofrc-qm9-bs120*/eval.log')]
    assert all([int(log[-1,0])==3000000 for log in logs]), "Incomplete training"
    MAE = [log[-1,2]*hartree2meV for log in logs]
    print("# QM9[@2014_RamakrishnanDraletal]\n")
    print(f"Energy MAE:  {np.mean(MAE):.2f}(std:{np.std(MAE):.2f}) meV.")

def report_md17():
    results = {}
    for logf in glob('models/pinet-pot-md17-uracil-bs120-*/eval.log'):
        mol = re.match(r"models\/.+md17-(.*)-\d\/eval.log", logf)[1]
        log = np.loadtxt(logf)
        e_mae = log[-1, 2]
        f_mae = log[-1,7]
        if mol in results:
            results[mol].append([e_mae, f_mae])
        else:
            results[mol] = [[e_mae, f_mae]]

    print("MD17[@2017_ChmielaTkatchenkoEtAl]\n")
    for k, v in sorted(results.items()):
        E_MAE = np.array(v)[:,0] * kcalpermol2meV
        F_MAE = np.array(v)[:,1] * kcalpermol2meV
        print(f"- {k+':':10s} Energy MAE={np.mean(E_MAE):.2f}(std:{np.std(E_MAE):.2f}) meV; "
              f"Force MAE={np.mean(F_MAE):.2f}(std:{np.std(F_MAE):.2f}) meV/Å.")
