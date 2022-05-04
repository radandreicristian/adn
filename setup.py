from setuptools import setup, find_packages

setup(
    name="adn_torch",
    packages=find_packages(),
    version="0.0.1",
    license="MIT",
    description="Attention Diffusion Network implementation in Pytorch",
    author="Andrei-Cristian Rad",
    author_email="radandreicristian@gmail.com",
    url="https://github.com/radandreicristian/adn",
    keywords=["attention", "artificial intelligence", "deep learning", "traffic"],
    install_requires=["torch"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
)
