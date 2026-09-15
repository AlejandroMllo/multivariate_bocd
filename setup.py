from setuptools import find_packages, setup

setup(
    name="multivariate-bocd",
    version="0.1.0",
    description="Multivariate Bayesian online changepoint detection with a Gaussian-Wishart predictive model.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Alejandro Murillo-Gonzalez",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"multivariate_bocd": ["data/*.csv"]},
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=["numpy>=1.23", "pandas>=1.5", "torch>=2.0"],
    extras_require={
        "examples": ["matplotlib>=3.7"],
        "dev": ["pytest>=7", "build>=1.0", "twine>=5.0"],
    },
    project_urls={
        "Homepage": "https://github.com/AlejandroMllo/multivariate_bocd",
        "Repository": "https://github.com/AlejandroMllo/multivariate_bocd",
        "Paper": "https://doi.org/10.1177/02783649261431863",
        "Related Source": "https://github.com/AlejandroMllo/situationally_aware_dynamics_learning",
    },
)
