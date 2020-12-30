

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pywedge",
    version="0.5.1.3",
    author="Venkatesh Rengarajan Muthu",
    author_email="taknev83@gmail.com",
    description="Makes interactive Charts, Interactive baseline, Interactive Hyperparameter Tuning & predictions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    url="https://github.com/taknev83/pywedge",
    packages=['pywedge'],
    include_package_data=True,
    install_requires=[
        "xgboost>1.2",
        "pandas>1.1.4",
        "catboost>0.24",
        "numpy>1.19.3",
        "scikit-learn>0.23",
        "imbalanced-learn>=0.70",
        "plotly>4.12",
        "ipywidgets>=7.5.1",
        "mlflow"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
