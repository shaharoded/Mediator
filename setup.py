from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements.txt
requirements_path = Path(__file__).parent / "requirements.txt"
install_requires = []
if requirements_path.exists():
    install_requires = [
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="mediator-kbta",
    version="1.0.0",
    author="Shahar Oded",
    author_email="your.email@example.com",
    description="Knowledge-Based Temporal Abstraction (KBTA) for clinical data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/Mediator",
    packages=find_packages(exclude=["unittests", "images"]),
    include_package_data=True,
    package_data={
        "core": ["knowledge-base/**/*.xml", "knowledge-base/**/*.xsd", "knowledge-base/**/*.json"],
        "backend": ["queries/**/*.sql"],
    },
    install_requires=install_requires,
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "mediator=core.mediator:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
