"""
Setup script for Advanced Stock Prediction System
"""

from setuptools import setup, find_packages

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="stock-prediction-system",
    version="1.0.0",
    author="SaiChen22",
    author_email="your-email@domain.com",
    description="Advanced Stock Prediction System with ML Parameter Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SaiChen22/stock-prediction",
    project_urls={
        "Bug Tracker": "https://github.com/SaiChen22/stock-prediction/issues",
        "Documentation": "https://github.com/SaiChen22/stock-prediction/docs",
        "Source Code": "https://github.com/SaiChen22/stock-prediction",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0", 
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "jupyter>=1.0.0",
            "notebook>=7.0.0"
        ],
        "plotting": [
            "matplotlib>=3.6.0",
            "seaborn>=0.12.0",
            "plotly>=5.0.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "stock-predict=demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords="stock prediction machine learning finance trading optimization bayesian genetic algorithm",
)