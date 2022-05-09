# Replication does not measure scientific productivity

## System Requirements

### Hardware
Our data and analysis were conducted on a linux server with sixty-four 2.6GHz CPUs and 256GB RAM. 
It has a 256GB primary disk partition on an RAID1 SSD and an 8TB secondary partition for home directories on a RAID6 storage array.
This is overkill and this could should run fine on any computer made in the last decade.  

### Software 
Running this software will requre you have Python 3.9.7 installed. 
This code using python and requires that you have python3 installed on your system. 
To install required packages, clone the repository and navigate to the directory. Then type: 

``
pip install -r requirements.txt
``
Installation time will vary depending upon what you already have installed, typically no more than 30 minutes. 

## Reproducing our analysis

### Downloading data

The data used in our analysis can be downloaded from the original study's OSF Repository: 

https://osf.io/fgjvw/download

## Running the code

The easiset way to run the code is to execute the jupyter notebooks from the command line:

``
jupyter nbconvert --to notebook --execute Theory.ipynb
jupyter nbconvert --to notebook --execute StatisticalModel.ipynb
``

<h2>Replicating the results</h2>
The jupyter notebooka reproduce all results as they are presented in the paper. 
