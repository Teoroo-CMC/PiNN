from setuptools import setup, find_packages

setup(name='pinn',
      version='0.1',
      description='Pair interaction neural network',
      url='https://github.com/yqshao/pinn',
      author='Yunqi Shao',
      author_email='yunqi_shao@yahoo.com',
      license='GPLv3',
      packages=find_packages(),
      install_requires=['ase', 'numpy', 'pyyaml'],
      entry_points = {
          'console_scripts': ['pinn_train=pinn.trainer:main']
      }
)
