import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    requirements = f.readlines()

setuptools.setup(
    name="fp_bnn",
    version="0.0.1",
    author="Georges Pantalos, Jonas Rothfuss",
    author_email="georgespantalos@gmail.com",
    description="Functional Priors for Bayesian Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={'fp_bnn': 'fp_bnn'},
    packages=setuptools.find_packages(),
    package_data={'': ['*.json']},
    install_requires=requirements,
)
