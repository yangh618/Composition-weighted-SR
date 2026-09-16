#!/usr/bin/env python3
"""Setup script for Composition-weighted Symbolic Regression (CWSR)."""

from pathlib import Path
from setuptools import setup, find_packages

# ---------------------------------------------------------------------------
# Read requirements from requirements.txt
# ---------------------------------------------------------------------------
def parse_requirements(filename: str) -> list[str]:
    """Return a list of package dependencies from a requirements.txt file."""
    reqs = []
    req_path = Path(__file__).parent / filename
    if req_path.exists():
        for line in req_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                reqs.append(line)
    return reqs


# ---------------------------------------------------------------------------
# Read long description from README if present
# ---------------------------------------------------------------------------
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text()

# ---------------------------------------------------------------------------
# Additional core dependencies not listed in requirements.txt but required
# by the Matbench loading path (datasets.matbench)
# ---------------------------------------------------------------------------
extra_deps = [
    "pymatgen",
    "matbench",
    "ase",
    "matplotlib",
    "numba",
]

setup(
    name="cwsr",
    version="0.2.0",
    description="Composition-weighted Symbolic Regression (CWSR) with MCTS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="",
    author_email="",
    url="",
    license="MIT",
    # -----------------------------------------------------------------------
    # Packages
    # -----------------------------------------------------------------------
    packages=find_packages(include=[
        "analysis",
        "analysis.*",
        "cwsr",
        "cwsr.*",
        "datasets",
        "datasets.*",
    ]),
    # -----------------------------------------------------------------------
    # Bundled databases shipped inside the datasets package
    # -----------------------------------------------------------------------
    package_data={
        "datasets": ["alloys/data/*.npz", "alloys/data/*.md"],
    },
    # -----------------------------------------------------------------------
    # Dependencies
    # -----------------------------------------------------------------------
    install_requires=parse_requirements("requirements.txt") + extra_deps,
    python_requires=">=3.10",
    # -----------------------------------------------------------------------
    # Optional development / test dependencies
    # -----------------------------------------------------------------------
    extras_require={
        "test": ["pytest>=7"],
    },
    # -----------------------------------------------------------------------
    # CLI entry points
    # -----------------------------------------------------------------------
    entry_points={
        "console_scripts": [
            "cwsr-query=cwsr.predict.query:main",
            "cwsr-gallery=cwsr.plotting.gallery:main",
            "cwsr-inverse=analysis.inverse:main",
            "cwsr-pareto=analysis.pareto:main",
            "cwsr-bootstrap=analysis.bootstrap:main",
        ],
    },
    # -----------------------------------------------------------------------
    # Include non-Python data files
    # -----------------------------------------------------------------------
    include_package_data=True,
    # -----------------------------------------------------------------------
    # Classifiers
    # -----------------------------------------------------------------------
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
)
