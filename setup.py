"""
Copyright (C) 2021 Rosalind Franklin Institute

"""


from setuptools import setup, find_packages


setup(
    version = '0.0',
    name = 'leopardgecko',
    description = 'Tools to analyse and process results from predicted segmentation of microscopy volumetric tomography data',
    url = 'https://github.com/rosalindfranklininstitute/LeopardGecko',
    author = 'Luis Perdigao',
    author_email='luis.perdigao@rfi.ac.uk',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    test_suite='tests',
    classifiers=[
        'Development Status :: ok ',
        'License :: Not decided yet',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
        'Operating System :: POSIX :: Linux',
    ],
    license='Not sure',
    zip_safe=False,
    install_requires=[
        'numpy>=1.18.0',
        'h5py',
        'pyyaml',
        'dask',
        'torch',
        'scipy'
    ]
)