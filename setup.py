from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name='ml',
    version='1.0',
    description='Functions for geostatistical machine learning',
    author='KetilH',
    author_email='kehok@equinor.com',
    packages=['ml'],  #same as name
    install_requires=[], # avoid reinstalling stuff
    # install_requires=required, #external packages as dependencies
    zip_safe=False,
)