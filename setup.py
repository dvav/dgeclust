from setuptools import setup

setup(name='DGEclust',
      version='14.08',
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
      author_email='dimitris.vavoulis@bristol.ac.uk',
      platforms=['UNIX'],
      license='MIT',
      packages=['dgeclust', 'dgeclust.models'],
      install_requires=[
          'numpy>=1.8',
          'scipy>=0.14',
          'pandas>=0.14',
          'matplotlib>=1.3',
          'ipython>=2.2'
      ],
      zip_safe=False)
