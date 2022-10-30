# Implementation Details

Implementation Details. We adopt Drain [3] to parse the log messages into
log keys. Regarding baselines, we leverage the package Loglizer [4] to evaluate
PCA, OCSVM, iForest as well as LogCluster for anomaly detection and adopt
the open source deep learning-based log analysis toolkit to evaluate DeepLog
and LogAnomaly 3
. For LogBERT, we construct a Transformer encoder by using two Transformer layers. The dimensions for the input representation and
hidden vectors are 50 and 256, respectively. The hyper-parameters, including α
in Equation 7, m the ratio of masked log keys for the MKLP task, r the number of predicted anomalous log keys, and g the size of top-g candidate set for
anomaly detection are tuned based on a small validation set. In our experiments,
both training and detection phases have the same ratio of masked log keys m.

# Upstream of `Predictor` and Other Abstractions

Major classes are modified based on https://github.com/donglee-afar/logdeep.

# Upstream Source of `bert_pytorch` Folder

`bert_pytorch` is directly vendoring from https://github.com/codertimo/BERT-pytorch, 
which is used for pretraining BERT, probably with reduced layer number to minimize training cost.

# Dataset Processing Code

## HDFS

### Input Files and Directory

```python
import torch

data_dir = os.path.expanduser("~/.dataset/hdfs")
```

### Output Files and Directory


```python
log_file   = "HDFS.log"  # The input log file name

log_structured_file = output_dir + log_file + "_structured.csv"
log_templates_file = output_dir + log_file + "_templates.csv"
log_sequence_file = output_dir + "hdfs_sequence.csv"
```


# LogBERT: Log Anomaly Detection via BERT   

### [ARXIV](https://arxiv.org/abs/2103.04475)   

This repository provides the implementation of Logbert for log anomaly detection. 
The process includes downloading raw data online, parsing logs into structured data, 
creating log sequences and finally modeling. 

<!-- ![alt](img/log_preprocess.png) -->

## Configuration
- Ubuntu 20.04
- NVIDIA driver 460.73.01 
- CUDA 11.2
- Python 3.8
- PyTorch 1.9.0

## Installation
This code requires the packages listed in requirements.txt.
An virtual environment is recommended to run this code

On macOS and Linux:  
```
python3 -m pip install --user virtualenv
python3 -m venv env
source env/bin/activate
pip install -r ./environment/requirements.txt
deactivate
```
Reference: https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/

An alternative is to create a conda environment:
```
    conda create -f ./environment/environment.yml
    conda activate logbert
```
Reference: https://docs.conda.io/en/latest/miniconda.html

## Experiment
Logbert and other baseline models are implemented on [HDFS](https://github.com/logpai/loghub/tree/master/HDFS), [BGL](https://github.com/logpai/loghub/tree/master/BGL), and [thunderbird]() datasets

### HDFS example
```shell script

cd HDFS

sh init.sh

# process data
python data_process.py

#run logbert
python logbert.py vocab
python logbert.py train
python logbert.py predict

#run deeplog
python deeplog.py vocab
# set options["vocab_size"] = <vocab output> above
python deeplog.py train
python deeplog.py predict 

#run loganomaly
python loganomaly.py vocab
# set options["vocab_size"] = <vocab output> above
python loganomaly.py train
python loganomaly.py predict

#run baselines

baselines.ipynb
```

### Folders created during execution
```shell script 
~/.dataset //Stores original datasets after downloading
project/output //Stores intermediate files and final results during execution
```
