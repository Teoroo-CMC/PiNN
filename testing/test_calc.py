from ase import atoms
from ase.optimize import BFGS
from ase.collections import g2
from pinn import PINN, pinn_model

def test_calculator():
    atoms = g2['CH4']
    atoms.calc = PINN('examples/l6-128.json')
    opt = BFGS(atoms)
    opt.run(fmax=0.02)
