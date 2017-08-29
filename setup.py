from setuptools import setup

setup(name='grmpy',
      version='0.1',
      description='grmpy is a package for simulation and analysis of economic aspects of the \
            Generalized Roy Model',
      url='https://github.com/grmToolbox/grmpy',
      author='Phillip Eisenhauer',
      author_email='eisenhauer@policy-lab.org',
      license='MIT',
      packages=['numpy', 'scipy', 'pytest', 'pandas', 'statsmodels'],
      zip_safe=False)
