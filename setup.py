from setuptools import setup

setup(name='grmpy',
      version='0.1',
      description='grmpy is a package for simulation and analysis of economic aspects of the \
            Generalized Roy Model',
      url='https://github.com/grmToolbox/grmpy',
      author='Philip Eisenhauer',
      author_email='eisenhauer@policy-lab.org',
      license='MIT',
      install_requires=['numpy', 'scipy', 'pytest', 'pandas', 'statsmodels'],
      zip_safe=False)
