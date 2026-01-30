from setuptools import find_packages, setup

setup(
    name="energy_trading",
    version="0.1.0",
    description="Energy trading machine learning project",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
)
