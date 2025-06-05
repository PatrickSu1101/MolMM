#!/bin/bash
set -e

python 000_run_multi.py -n CNN_pH_kfold_val_pure -d results/baseline_sub -c1 ckpt/baseline_sub -l 1 -t substrates_classes

python 000_run_multi.py -n CNN_pH_kfold_nval_pure -d results/baseline_inh -c1 ckpt/baseline_inh -l 1 -t inhibitors_classes

python 000_run_multi.py -n CNN_pH_kfold_val_stop -d results/baseline_sub -c1 ckpt/baseline_sub -l 1 -t substrates_classes
python 000_run_multi.py -n CNN_pH_kfold_val_full -d results/baseline_sub -c1 ckpt/baseline_sub -l 1 -t substrates_classes
python 000_run_multi.py -n CNN_pH_kfold_val_freeze -d results/baseline_sub -c1 ckpt/baseline_sub -l 1 -t substrates_classes

python 000_run_multi.py -n CNN_pH_kfold_nval_stop -d results/baseline_inh -c1 ckpt/baseline_inh -l 1 -t inhibitors_classes
python 000_run_multi.py -n CNN_pH_kfold_nval_full -d results/baseline_inh -c1 ckpt/baseline_inh -l 1 -t inhibitors_classes
python 000_run_multi.py -n CNN_pH_kfold_nval_freeze -d results/baseline_inh -c1 ckpt/baseline_inh -l 1 -t inhibitors_classes

python 000_run_multi.py -n CNN_pH_kfold_val_merge_stop -d results/baseline_sub -c1 ckpt/baseline_sub -l 1 -t substrates_classes
python 000_run_multi.py -n CNN_pH_kfold_val_merge_full -d results/baseline_sub -c1 ckpt/baseline_sub -l 1 -t substrates_classes
python 000_run_multi.py -n CNN_pH_kfold_val_merge_freeze -d results/baseline_sub -c1 ckpt/baseline_sub -l 1 -t substrates_classes

python 000_run_multi.py -n CNN_pH_kfold_nval_merge_stop -d results/baseline_inh -c1 ckpt/baseline_inh -l 1 -t inhibitors_classes
python 000_run_multi.py -n CNN_pH_kfold_nval_merge_full -d results/baseline_inh -c1 ckpt/baseline_inh -l 1 -t inhibitors_classes
python 000_run_multi.py -n CNN_pH_kfold_nval_merge_freeze -d results/baseline_inh -c1 ckpt/baseline_inh -l 1 -t inhibitors_classes
