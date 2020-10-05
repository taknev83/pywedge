# -*- coding: utf-8 -*-

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pywedge",
    version="0.1",
    author="Venkatesh rengarajan Muthu",
    author_email="taknev83@gmail.com",
    description="Cleans raw data, runs baseline models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='Apache License 2.0',
    url="https://github.com/taknev83/pywedge/blob/main/pywedge.py",
    packages=setuptools.find_packages(exclude=("tests",)),
    install_requires=[
        "jupyter",
        "xgboost>=1.1.1",
        "pandas",
        "scikit-learn>=0.23.1",
        "imbalanced-learn>=0.7",
    	"featuretools",        
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
