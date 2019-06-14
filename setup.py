from setuptools import setup
from setuptools import find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="DeepXDE",
    version="0.1.1",
    description="Deep learning library for solving differential equations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Lu Lu",
    author_email="lululxvi@gmail.com",
    url="https://github.com/lululxvi/deepxde",
    download_url="https://github.com/lululxvi/deepxde/tarball/v0.1.1",
    license="Apache-2.0",
    install_requires=[
        "matplotlib",
        "numpy",
        "salib",
        "scikit-learn",
        "scipy",
        "tensorflow",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "Deep Learning",
        "Machine Learning",
        "Neural Networks",
        "Scientific computing",
        "Differential equations",
        "PDE solver",
    ],
    packages=find_packages(),
    include_package_data=True,
)
