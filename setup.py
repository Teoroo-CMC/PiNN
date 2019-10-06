from setuptools import setup, find_packages

setup(name='pinn',
      version='dev',
      description='Pair interaction neural network',
      url='https://github.com/yqshao/pinn',
      author='Yunqi Shao',
      author_email='yunqi_shao@yahoo.com',
      license='BSD',
      packages=find_packages(),
      install_requires=['ase', 'numpy', 'pyyaml'],
      entry_points={
          'console_scripts': ['pinn_train=pinn.trainer:main']
      }
)
