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
# by the dataloader / eval modules
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
    version="1.0.0",
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
    packages=find_packages(include=["iMCTS", "iMCTS.*", "utils", "utils.*"]),
    # -----------------------------------------------------------------------
    # Stand-alone modules (not inside a package directory)
    # -----------------------------------------------------------------------
    py_modules=[
        "dataloader",
        "eval",
        "visiualize",
        "run_cwsr",
    ],
    # -----------------------------------------------------------------------
    # Dependencies
    # -----------------------------------------------------------------------
    install_requires=parse_requirements("requirements.txt") + extra_deps,
    python_requires=">=3.10",
    # -----------------------------------------------------------------------
    # CLI entry points
    # -----------------------------------------------------------------------
    entry_points={
        "console_scripts": [
            "cwsr=run_cwsr:main",
            "cwsr-eval=eval:main",
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
