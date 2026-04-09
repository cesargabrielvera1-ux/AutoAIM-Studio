"""Setup script for Materials AutoML Studio."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Materials AutoML Studio - Automated Machine Learning for Materials Science"

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]
else:
    requirements = [
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "torch>=2.0.0",
        "PyQt6>=6.5.0",
    ]

setup(
    name="materials-automl",
    version="1.0.0",
    author="Materials AutoML Team",
    description="Automated Machine Learning for Materials Science",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/materials_automl",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Materials Science",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "gpu": ["cuda-python"],
        "all": [
            "xgboost>=2.0.0",
            "lightgbm>=4.0.0",
            "catboost>=1.2.0",
            "optuna>=3.4.0",
            "shap>=0.42.0",
            "matminer>=0.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "materials-automl=app.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
