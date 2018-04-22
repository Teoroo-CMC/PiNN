from setuptools import setup

setup(name='pinn',
      version='0.1',
      description='Pair interaction neural network',
      url='https://github.com/yqshao/pinn',
      author='Yunqi Shao',
      author_email='yunqi_shao@yahoo.com',
      license='GPLv3',
      packages=['pinn'],
      install_requires=[
          'ase',
          'numpy',
          'h5py',
          'matplotlib',
          'tensorflow',
      ],
)
