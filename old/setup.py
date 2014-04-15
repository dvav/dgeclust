try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Non-parametric Bayesian clustering of digital gene expression data',
    'author': 'Dimitrios V. Vavoulis',
    'url': 'https://bitbucket.org/DimitrisVavoulis/dgeclust',
    'download_url': 'https://bitbucket.org/DimitrisVavoulis/dgeclust',
    'author_email': 'Dimitris.Vavoulis@bristol.ac.uk',
    'version': '0.1',
    'install_requires': ['nose'],
    'packages': ['DGEclust'],
    'scripts': [],
    'name': 'DGEclust'
}

setup(**config)