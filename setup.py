from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="trustworthy-ran-orchestration",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@university.edu",
    description="Trustworthy AI for Open RAN Orchestration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/trustworthy-ran-orchestration",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9,<3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.2",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
    },
)
