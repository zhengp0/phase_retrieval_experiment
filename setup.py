from setuptools import setup
from setuptools import find_packages


setup(name='phre',
      version='0.0.0',
      description='Phase retrieval experiment.',
      url='https://github.com/zhengp0/phase_retrieval_experiment',
      author='Peng Zheng',
      author_email='zhengp@uw.edu',
      license='MIT',
      packages=find_packages(where='src'),
      package_dir={'': 'src'},
      install_requires=['numpy',
                        'scipy',
                        'matplotlib',
                        'pytest'],
      zip_safe=False)
