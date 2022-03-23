import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="meta_learn",
    version="0.0.1",
    author="Jonas Rothfuss",
    author_email="jonas.rothfuss@gmail.com",
    description="Bayesian Optimization with meta-learned Gaussian Processes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={'meta_bo': 'meta_bo'},
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'gpytorch',
        'absl-py',
        'pandas',
        'scipy',
        'matplotlib',
        'absl-py'
    ],
)