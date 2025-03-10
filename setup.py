#!/usr/bin/env python
"""
Setup script for Neural Carnival.
"""

from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Read long description from README.md
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="neural_carnival",
    version="1.0.0",
    description="A sophisticated neural network simulation and visualization system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Neural Carnival Team",
    author_email="info@neuralcarnival.example.com",
    url="https://github.com/yourusername/neuralCarnival",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "neural-carnival=run_simulation:main",
        ],
    },
) 