from setuptools import setup, find_packages
import os

# Read the contents of README.md
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Read the requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="cities-skylines-2-agent",
    version="0.1.0",
    description="Autonomous reinforcement learning agent for Cities: Skylines 2",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Project Contributors",
    author_email="your-email@example.com",
    url="https://github.com/yourusername/cities-skylines-2-agent",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "cs2-train=src.train:main",
            "cs2-run-env=scripts.run_environment:main",
        ],
    },
) 