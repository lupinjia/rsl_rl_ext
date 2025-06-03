from setuptools import setup, find_packages

setup(name='rsl_rl_ext',
      version='0.1.0',
      author='Yasen Jia',
      author_email='jason_1120202397@163.com',
      license="BSD-3-Clause",
      packages=find_packages(),
      description='Extension for rsl_rl',
      python_requires='>=3.6',
      install_requires=[
            "torch>=1.4.0",
            "torchvision>=0.5.0",
            "numpy>=1.16.4",
            "tqdm>=4.32.2",
      ],
      )
