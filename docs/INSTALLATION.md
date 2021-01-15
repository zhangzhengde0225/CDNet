# Installation

This document describes how to get CNDet source code and configure the running environment.

## Requirements
+ Python 3.6+
+ Pytorch 1.6+
+ Linux or MacOS

## Get CDNet Code

##### use git:
```python
git clone https://github.com/zhangzhengde0225/CDNet.git
```
##### from github web:
Browse the [CDnet] repository and Click "Code"-"Download ZIP".

## Configure Environment
[anaconda](https://www.anaconda.com) is recommanded.

Haven't Anaconda been installed yet? Download anaconda installer [here](https://www.anaconda.com/products/individual#Downloads).
```python
chmod +x ./Anaconda3-2020.11-Linux-.sh
./Anaconda3-2020.11-Linux-.sh  # install
which conda  # verify installation
```

After having CONDA, directly import the fully configured environment:
```python
conda env creat -f conda_cdnet_env.yaml
```

or Creating a environment from sratch:
```python
conda create --name cdnet python=3.7  # create a env named as "cdnet"
conda activate cdnet  # activate the env
which pip  # verify pip 
pip install -r requirements.txt  # install packages use pip
# or use conda to install package
conda install <PACKAGE>
```
 



