# Implementation Details

Implementation Details. We adopt Drain [3] to parse the log messages into
log keys. Regarding baselines, we leverage the package Loglizer [4] to evaluate
PCA, OCSVM, iForest as well as LogCluster for anomaly detection and adopt
the open source deep learning-based log analysis toolkit to evaluate DeepLog
and LogAnomaly. For LogBERT, we construct a Transformer encoder by using two Transformer layers. The dimensions for the input representation and hidden vectors are 50 and 256, respectively. The hyper-parameters, including α
in Equation 7, $m$ the ratio of masked log keys for the MKLP task, $r$ the number of predicted anomalous log keys, and $g$ the size of top-$g$ candidate set for anomaly detection are tuned based on a small validation set. In our experiments,
both training and detection phases have the same ratio of masked log keys $m$.

# Upstream of `Predictor` and Other Abstractions

Major classes are modified based on https://github.com/donglee-afar/logdeep.

# Upstream Source of `bert_pytorch` Folder

`bert_pytorch` is directly vendoring from https://github.com/codertimo/BERT-pytorch, 
which is used for pretraining BERT, probably with reduced layer number to minimize training cost.

# Dataset Processing Code

## Details on HDFS Dataset

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

## Details on BGL Dataset

Details on BGL Dataset

```python
# In the first column of the log, "-" indicates non-alert messages while others are alert messages.
def count_anomaly():
    total_size = 0
    normal_size = 0
    with open(data_dir + log_file, encoding="utf8") as f:
        for line in f:
            total_size += 1
            if line.split(' ',1)[0] == '-':
                normal_size += 1
    print("total size {}, abnormal size {}".format(total_size, total_size - normal_size))
```

## Details on TBird Dataset

TBird dataset contains more than 30GB logs, which may require some skills to process.

```python
data_dir = os.path.expanduser("~/.dataset/tbird/")
output_dir = "../output/tbird/"
raw_log_file = "Thunderbird.log"
sample_log_file = "Thunderbird_20M.log"
sample_window_size = 2*10**7
sample_step_size = 10**4
window_name = ''
log_file = sample_log_file

parser_type = 'drain'
#mins
window_size = 1
step_size = 0.5
train_ratio = 6000

########
# count anomaly
########
# count_anomaly(data_dir + log_file)
# sys.exit()

#########
# sample raw data
#########
sample_raw_data(data_dir+raw_log_file, data_dir+sample_log_file, sample_window_size, sample_step_size)
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

```shell
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

```shell
~/.dataset //Stores original datasets after downloading
project/output //Stores intermediate files and final results during execution
```
