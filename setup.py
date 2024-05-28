#!/usr/bin/env python
from setuptools import find_namespace_packages, setup

setup(
    packages=find_namespace_packages(
        include=["langchain_iris", "langchain_iris.*"]
    ),
    install_requires=[
        "langchain>=0.0.348",
        "langchain-community>=0.2.1",
        "sqlalchemy-iris>=0.14.0",
    ],
    python_requires=">3.7",
)
