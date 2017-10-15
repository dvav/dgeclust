from setuptools import setup

setup(name='DGEclust',
      version='17.10.16',
      description='Hierarchical non-parametric Bayesian clustering of digital expression data',
      long_description=open('README.md').read(),
      classifiers=[
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 2.7',
          'Operating System :: Unix',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
      ],
      url='http://dvav.me/dgeclust/',
      author='Dimitrios V. Vavoulis',
      author_email='dimitris.vavoulis@well.ox.ac.uk',
      platforms=['UNIX'],
      license='MIT',
      packages=['dgeclust', 'dgeclust.models'],
      install_requires=[
          'numpy>=1.13',
          'scipy>=0.19',
          'pandas>=0.20',
          'matplotlib>=2.1',
          'ipython>=5.4'
      ],
      zip_safe=False)
