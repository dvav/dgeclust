from setuptools import setup

setup(name='DGEclust',
      version='14.08a',
      description='Hierarchical non-parametric Bayesian clustering of digital expression data',
      long_description=open('README.md').read(),
      classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
      ],
      url='http://dvav.me/dgeclust/',
      download_url='',
      author='Dimitrios V. Vavoulis',
      author_email='dimitris.vavoulis@bristol.ac.uk',
      license='LICENSE.md',
      packages=['dgeclust', 'dgeclust.gibbs', 'dgeclust.models'],
      scripts=['bin/clust', 'bin/pvals', 'bin/simmat', 'bin/_clust.py', 'bin/_pvals.py', 'bin/_simmat.py'],
      install_requires=[
          'numpy',
          'scipy',
          'pandas',
          'matplotlib'
      ],
      include_package_data=True,
      zip_safe=False)
