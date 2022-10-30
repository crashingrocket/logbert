#!/usr/bin/env bash
cd HDFS

sh init.sh

# process data
python data_process.py

#run logbert
python loganomaly.py vocab
python loganomaly.py train
python loganomaly.py predict