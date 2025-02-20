from setuptools import setup, find_packages

setup(
    name='Trinoculars',
    version='1.0.0',
    packages=find_packages(),
    url='https://github.com/CoffeBank/Trinoculars',
    license=open("LICENSE.md", "r", encoding="utf-8").read(),
    author='',
    author_email='',
    description='An improved version of the Binoculars language model text detector for ru datasets.',
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    install_requires=open("requirements.txt", "r", encoding="utf-8").read().splitlines(),
)
