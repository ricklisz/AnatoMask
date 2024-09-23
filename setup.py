import setuptools

if __name__ == "__main__":
    setuptools.setup(
        name="nnunetv2",  # Replace with your project name
        version="0.1",
        packages=setuptools.find_packages(exclude=['figs']),  # Exclude the figs folder
        install_requires=[
            # List your dependencies here, for example:
            # "numpy", "torch", etc.
        ],
    )

