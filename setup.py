from setuptools import setup, find_packages

setup(
    name='anatomask',
    version='0.1',
    packages=find_packages(include=['nnunetv2', 'nnunetv2.*', 'ssl_pretrain', 'ssl_pretrain.*']),
    install_requires=[
        'torch',  # Add other dependencies as needed
        'numpy',
        'scipy',
        # Include all other dependencies here
    ],
)
