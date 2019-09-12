# Recommended Installation via Conda environment

## Prerequisites

Make sure you have [Conda](https://docs.anaconda.com/anaconda/install/) installed on your machine.

## Setting up the Conda environment

Create an environment for SCHISM using the provided [environment yaml file](https://github.com/KarchinLab/SCHISM/blob/master/schism-env-macos.yaml)

```
conda env create -f schism-env-macos.yaml
```

## Install SCHISM within the Conda environment

Download the latest SCHISM release

```
wget https://github.com/KarchinLab/SCHISM/archive/SCHISM-1.1.3.tar.gz
```

Install SCHISM within the Conda environment

```
source activate schism-orig
pip install SCHISM-1.1.3.tar.gz
```
