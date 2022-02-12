import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="comppyss",
    version="0.0.2",
    author="Andrew Sung",
    author_email="aysung300@gmail.com",
    description="Python implementation of the CompPASS algorithm for scoring AE-MS experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AYSung/comppyss/",
    project_urls={
        "Bug Tracker": "https://github.com/AYSung/comppyss/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "comppyss"},
    packages=setuptools.find_packages(where="comppyss"),
    python_requires=">=3.10",
)
