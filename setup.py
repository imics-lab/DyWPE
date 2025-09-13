from setuptools import setup, find_packages

setup(
    name="dywpe",
    version="1.0.0",
    author="Habib Irani, Vangelis Metsis",
    description="Dynamic Wavelet Positional Encoding for Time Series Transformers",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=1.0.0",
    ],
)