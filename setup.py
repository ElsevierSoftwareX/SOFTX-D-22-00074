from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = "dpivsoft",
    version="0.2.5",
    url="https://gitlab.com/jacabello/dpivsoft_python",
    author = "Jorge Aguilar-Cabello",
    python_requires=">=3.7",
    packages = find_packages(),
    include_package_data = True,
    long_description = long_description,
    long_description_content_type = "text/markdown",
    setup_requires=[
        'setuptools',
    ],
    install_requires = [
        'numpy>=1.2',
        'reikna>=0.7.6',
        'scipy>=1.5',
        'opencv-python>=4.5',
        'matplotlib>=3',
        'PyYAML>=5.4',
        'pyopencl>=2021',
        'Shapely>=1.7',
        'importlib_resources>=5',
    ],
    extras_require = {
        "dev": [
            "pytest>=3.7",
        ],
    },
    classifiers=[
         # Sublist of all supported Python versions.
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',

        #Miscelaneous metadata
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
    ],
)
