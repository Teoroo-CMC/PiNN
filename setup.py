import os
from setuptools import setup, find_packages

on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    INSTALL_REQUIRES = ['ase', 'numpy', 'pyyaml', 'tensorflow-cpu==1.15']
else:
    INSTALL_REQUIRES = ['ase', 'numpy', 'pyyaml', 'tensorflow==1.15']

setup(name='pinn',
      version='dev',
      description='Pair interaction neural network',
      url='https://github.com/yqshao/pinn',
      author='Yunqi Shao',
      author_email='yunqi_shao@yahoo.com',
      license='BSD',
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      entry_points={
          'console_scripts': ['pinn_train=pinn.trainer:main']
      }
)
