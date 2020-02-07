#!/bin/bash

python3 main.py spam.data 0.1 2000 101
python3 main.py spam.data 0.4 2000 102
python3 main.py spam.data 10 100 104

python3 main.py SAheart.data 0.01 800 3
python3 main.py SAheart.data 0.2 200 4
python3 main.py SAheart.data 0.1 200 60

python3 main.py zip.train 4 100 60
python3 main.py zip.train 4 100 20
python3 main.py zip.train 3 100 30

python3 roc_curve_grapher.py ./roc-data/spam.data_step_0.1_itr_2000_seed_101_roc_data.npy ./roc-data/spam.data_step_0.4_itr_2000_seed_102_roc_data.npy ./roc-data/spam.data_step_10.0_itr_100_seed_104_roc_data.npy

python3 roc_curve_grapher.py ./roc-data/SAheart.data_step_0.01_itr_800_seed_3_roc_data.npy ./roc-data/SAheart.data_step_0.2_itr_200_seed_4_roc_data.npy ./roc-data/SAheart.data_step_0.1_itr_200_seed_60_roc_data.npy

python3 roc_curve_grapher.py ./roc-data/zip.train_step_4.0_itr_100_seed_60_roc_data.npy ./roc-data/zip.train_step_4.0_itr_100_seed_20_roc_data.npy ./roc-data/zip.train_step_3.0_itr_100_seed_30_roc_data.npy 
