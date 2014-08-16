from setuptools import setup

setup(name='DGEclust',
      version='14.08a',
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
      download_url='',
      author='Dimitrios V. Vavoulis',
      author_email='dimitris.vavoulis@bristol.ac.uk',
      platforms=['UNIX'],
      license='MIT',
      packages=['dgeclust', 'dgeclust.gibbs', 'dgeclust.models'],
      package_data={'dgeclust': ['config.json']},
      scripts=['bin/clust.py', 'bin/pvals.py', 'bin/simmat.py'],
      install_requires=[
          'numpy>=1.8',
          'scipy>=0.14',
          'pandas>=0.14',
          'matplotlib>=1.3'
      ],
      zip_safe=False)
