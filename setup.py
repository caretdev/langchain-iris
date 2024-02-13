#!/usr/bin/env python
from setuptools import find_namespace_packages, setup

setup(
    packages=find_namespace_packages(
        include=["langchain_iris", "langchain_iris.*"]
    ),
    install_requires=[
        "langchain==0.0.348",
        "sqlalchemy-iris>=0.13.0",
    ],
    python_requires=">3.7,<3.12",
)
