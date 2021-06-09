import setuptools

with open("README.md", "r") as file:
    long_description = file.read()

setuptools.setup(
    # id
    name="BraAndKet",
    version="0.7.2",

    # info
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/ZhengKeli/BraAndKet",
    description="BraAndKet is a library for numeral calculations of discrete quantum systems.",
    long_description=long_description,
    long_description_content_type="text/markdown",

    # author
    author="Zheng Keli",
    author_email="zhengkeli2009@126.com",

    # content
    packages=setuptools.find_packages(),

    # dependency
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'tqdm'
    ],
    extras_require={
        "sparse matrix": ['scipy'],
    },
)
