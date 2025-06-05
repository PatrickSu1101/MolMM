#!/bin/bash
set -e

# rm -rf results/baseline_*/DNN* ckpt/baseline_*/DNN*
python 000_run_multi.py -n DNN_pH_kfold_val_pure -d results/baseline_sub -c1 ckpt/baseline_sub -l 1 -t substrates_classes
python 000_run_multi.py -n DNN_pH_kfold_nval_pure -d results/baseline_inh -c1 ckpt/baseline_inh -l 1 -t inhibitors_classes