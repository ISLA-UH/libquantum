# libquantum installation
This section covers the basics on how to install and upgrade the libquntum library.

### Installing and upgrading liquantum with pip

The libquantum library can be installed using [pip](https://pip.pypa.io/en/stable/). The pip distribution and current RedPandas 
version can be found at [PyPI libquantum](https://pypi.org/project/libquantum/).

The following command can be used to install and/or upgrade libquantum:

```shell script
pip install libquantum --upgrade
```

### Verifying the installation

To check if the libquantum library has been installed correctly:
```shell script
pip show libquantum
```
The terminal should return the name of the module, version, summary, home-page, author, author email, license, location of the 
installed module and requires. An example is shown below:

```shell script
Name: libquantum
Version: 1.1.2
Summary: Library for implementing standardized time-frequency representations.
Home-page: https://github.com/RedVoxInc/libquantum
Author: RedVox
Author-email: dev@redvoxsound.com
License: Apache
Location: /path/where/module/is/installed
Requires: numpy, pandas, librosa, scipy, matplotlib, libwwz, redvox
Required-by: redvox-pandas
```

Return to _[main page](https://github.com/RedVoxInc/libquantum)_.