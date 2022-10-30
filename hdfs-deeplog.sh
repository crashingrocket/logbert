#!/usr/bin/env bash
cd HDFS

sh init.sh

# process data
python data_process.py

#run logbert
python deeplog.py vocab
python deeplog.py train
python deeplog.py predict