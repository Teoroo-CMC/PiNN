import os, re
from setuptools import setup, find_packages

with open('pinn/__init__.py') as f:
    version = re.search("__version__ = '(.*)'", f.read()).group(1)

def parse_reqs(filename):
    with open(filename) as f:
        return f.read().splitlines()

setup(name='pinn',
      version=version,
      description='Pair interaction neural network',
      url='https://github.com/yqshao/pinn',
      author='Yunqi Shao',
      author_email='yunqi_shao@yahoo.com',
      packages=find_packages(),
      classifiers=['License :: OSI Approved :: BSD License'],
      install_requires=['numpy>=1.15.0',
                        'ase>=3.19.0',
                        'click>=7.0',
                        'pyyaml>=3.01'],
      python_requires='>=3.6',
      extras_require={'cpu': ['tensorflow-cpu>=2.4'],
                      'gpu': ['tensorflow>=2.4'],
                      'dev': parse_reqs('requirements-dev.txt'),
                      'extra': parse_reqs('requirements-extra.txt')},
      entry_points={'console_scripts':
                    ['pinn=pinn.cli:main']}
)
