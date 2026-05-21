# Multi-Factor Selection System Python Package
# setup.py

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="multi-factor-selection",
    version="1.0.0",
    author="huyijiang",
    author_email="your-email@example.com",
    description="多因子选股系统 - Multi-factor stock selection system for Chinese A-share market",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/multi_factor_selection",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Natural Language :: Chinese (Simplified)",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "akshare>=1.18.0",
        "tushare>=1.2.0",
        "tqdm>=4.60.0",
        "loguru>=0.6.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mfs=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "config/*.yml"],
    },
)
