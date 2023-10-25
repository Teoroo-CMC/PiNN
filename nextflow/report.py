#!/usr/bin/env python

"""
Script to generate reports from benchmar results
"""
import re
import numpy as np
from glob import glob

hartree2meV = 27.2114079527 * 1000
kcalpermol2meV = 0.0433641153087705 * 1000

def report_qm9():
    logs = [np.loadtxt(logf) for logf in glob('models/*qm9*/eval.log')]
    assert all([int(log[-1,0])==3000000 for log in logs]), "Incomplete training"
    MAE = [log[-1,2]*hartree2meV for log in logs]
    print("# QM9[@2014_RamakrishnanDraletal]\n")
    print(f"Energy MAE:  {np.mean(MAE):.2f}(std:{np.std(MAE):.2f}) meV.")

def report_md17():
    results = {}
    for logf in glob('models/*md17*/eval.log'):
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

def main():
    report_qm9()
    report_md17()

if __name__ == "__main__":
    main()
