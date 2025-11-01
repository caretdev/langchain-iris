#!/usr/bin/env python
from setuptools import find_namespace_packages, setup

setup(
    packages=find_namespace_packages(
        include=["langchain_iris", "langchain_iris.*"]
    ),
    install_requires=[
        "langchain",
        "langchain-community",
        "sqlalchemy-iris>=0.18.0",
    ],
    python_requires=">3.8",
)
