#!/bin/bash

set -e

python 000_run_multi.py -n CNN_pH_kfold_val_multi_stop -d results/multitask_sub -c1 ckpt/multitask_sub -l 1 -t substrates_refine substrates_classes
python 000_run_multi.py -n CNN_pH_kfold_val_multi_freeze -d results/multitask_sub -c1 ckpt/multitask_sub -l 1 -t substrates_refine substrates_classes
python 000_run_multi.py -n CNN_pH_kfold_val_multi_full -d results/multitask_sub -c1 ckpt/multitask_sub -l 1 -t substrates_refine substrates_classes

python 000_run_multi.py -n CNN_pH_kfold_nval_multi_stop -d results/multitask_inh -c1 ckpt/multitask_inh -l 1 -t inhibitors_refine inhibitors_classes
python 000_run_multi.py -n CNN_pH_kfold_nval_multi_full -d results/multitask_inh -c1 ckpt/multitask_inh -l 1 -t inhibitors_refine inhibitors_classes
python 000_run_multi.py -n CNN_pH_kfold_nval_multi_freeze -d results/multitask_inh -c1 ckpt/multitask_inh -l 1 -t inhibitors_refine inhibitors_classes
