#!/usr/bin/env python
from setuptools import find_namespace_packages, setup
import os

thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + "/requirements.txt"

requirements = []
if os.path.isfile(requirementPath):
    with open("./requirements.txt") as f:
        for line in f.read().splitlines():
            requirements.append(line)

setup(
    packages=find_namespace_packages(
        include=["langchain-iris", "langchain-iris.*"]
    ),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">3.7,<3.12",
)
