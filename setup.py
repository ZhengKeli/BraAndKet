import setuptools

with open("README.md", "r") as file:
    long_description = file.read()

setuptools.setup(
    name="BraAndKet",
    version="0.4.0",
    author="Zheng Keli",
    author_email="zhengkeli2009@126.com",
    description="A library for a convenient representation of discrete quantum systems and their evolution.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ZhengKeli/BraAndKet",
    packages=setuptools.find_packages(exclude='./test'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
