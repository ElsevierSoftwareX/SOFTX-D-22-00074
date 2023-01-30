# DPIVSoft python

DPIVSoft is an open PIV algorithm developed by Thomas Leweke and Patrice Munier. The original source of DPIVSoft for Matlab enviorement is accesible at the following link: <https://www.irphe.fr/~meunier/>. This project tries to migrate the original source to a python enviorement and also add the capabilites of GPU computing using OpenCL.

DPIVSoft consist in Python and OpenCL modules for scripting and executing the analysis of 2D PIV image pairs. At current state some Python knowledge is needed to use this software. A graphic user interface is planned to be added to make the software more accesible to people without programming skills.

## Warning

The DPIVSoft Python version is still in it's *beta* state. This means that it can have some bugs and the API may change. However, testing and contributing is very welcome.

## Installing
Note: DPIVSoft is only compatible with Python 3.7, 3.8 & 3.9

### Using PYPI
DPIVSoft can be installed using PyPI from: <https://pypi.org/project/dpivsoft/>. In order to install just need to use the following command line:

```bash
$ pip install dpivsoft
```

### Build from source
Alternatively DPIVSoft can ge installed from source. In orther to do that, be sure you already have installed the package setuptools, otherwise install it using:

```bash
$ pip install setuptools
```

Clone the GitLab reposistory using:

```bash
$ git clone https://gitlab.com/jacabello/dpivsoft_python.git
```

For the global installation, first step is go to the source folder and generate an installable using the following command:

```bash
$ python setup.py sdist
```

If everithing is fine, a folder named "dist" must have been created. Install the installable generated inside "dist" folder by using:

```bash
$ pip install Installable_Name
```

## Learning to use DPIVSoft
There are tutorials and examples of how to compute PIV included with this package on source_folder/Examples

# Developing DPIVSOFT

Note: In this early state, there is not any test available yet.

To install dpivsoft, along with the tools that you need to develop and run test, run the following line in your virtualenv:

```bash
$ pip install -e.[dev]
```


## Contributors

1. [Jorge Aguilar-Cabello](https://gitlab.com/jacabello)

## Akcnowledgment

1. Patrice Meunier
2. Thomas Leweke
3. [Raul Infante-Sainz](https://gitlab.com/infantesainz)
4. [Luis Parras](https://gitlab.com/lparras)
5. Carlos del Pino

## How to cite this work
[https://doi.org/10.1016/j.softx.2022.101256](https://www.softxjournal.com/article/S2352-7110(22)00174-1/fulltext)
