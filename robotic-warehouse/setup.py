import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="rware",
    version="2.0.0",
    description="Multi-Robot Warehouse environment for reinforcement learning",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Filippos Christianos",
    url="https://github.com/semitable/robotic-warehouse",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    classifiers=[
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.7",
    ],
    install_requires=[
        "numpy",
        "gymnasium",
        "pyglet<2",
        "networkx",
        "six",
    ],
    extras_require={"test": ["pytest"]},
    include_package_data=True,
)
