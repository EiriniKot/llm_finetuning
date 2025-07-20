from setuptools import setup, find_packages

setup(
    name='llm-package-milestone',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'datasets==2.12.0',
        'transformers==4.53.2',
        'torch==2.7.1'
    ]
)