import os, re
from setuptools import setup, find_packages

with open('pinn/__init__.py') as f:
    version = re.search("__version__ = '(.*)'", f.read()).group(1)

setup(name='pinn',
      version=version,
      description='Pair interaction neural network',
      url='https://github.com/yqshao/pinn',
      author='Yunqi Shao',
      author_email='yunqi_shao@yahoo.com',
      license='BSD',
      packages=find_packages(),
      install_requires=['ase', 'numpy', 'pyyaml', 'tensorflow==1.15.2'],
      entry_points={
          'console_scripts': ['pinn_train=pinn.trainer:main']
      }
)
