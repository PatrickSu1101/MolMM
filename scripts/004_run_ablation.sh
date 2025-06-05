#!/bin/bash

set -e
# rm -rf results/meta* ckpt/meta*
# rm -rf results/meta_sub* ckpt/meta_sub*

# traditional
python 000_run_multi.py -n CNN_pH_kfold_val_meta_stop -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes
python 000_run_multi.py -n CNN_pH_kfold_val_meta_full -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes
python 000_run_multi.py -n CNN_pH_kfold_val_meta_freeze -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes

python 000_run_multi.py -n CNN_pH_kfold_nval_meta_stop -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes
python 000_run_multi.py -n CNN_pH_kfold_nval_meta_full -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes
python 000_run_multi.py -n CNN_pH_kfold_nval_meta_freeze -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes

python 000_run_multi.py -n CNN_pH_kfold_nval_meta_multi_stop -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_fda inhibitors_classes
python 000_run_multi.py -n CNN_pH_kfold_nval_meta_multi_freeze -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_fda inhibitors_classes
python 000_run_multi.py -n CNN_pH_kfold_nval_meta_multi_full -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_fda inhibitors_classes

# best model
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_stop -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_full -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_freeze -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_stop -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_full -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_freeze -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes

# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_inv_stop -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_inv_full -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_inv_freeze -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_inv_stop -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_inv_full -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_inv_freeze -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes

# python 000_run_multi.py -n CNN_pH_kfold_val_meta_guide_ab_stop -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes
# python 000_run_multi.py -n CNN_pH_kfold_val_meta_guide_ab_full -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes
# python 000_run_multi.py -n CNN_pH_kfold_val_meta_guide_ab_freeze -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes

# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_ab_stop -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_ab_full -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_ab_freeze -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes

python 000_run_multi.py -n CNN_pH_kfold_val_meta_guide_inv_ab_stop -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes
python 000_run_multi.py -n CNN_pH_kfold_val_meta_guide_inv_ab_full -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes
python 000_run_multi.py -n CNN_pH_kfold_val_meta_guide_inv_ab_freeze -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes

python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_inv_ab_stop -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes
python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_inv_ab_full -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes
python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_inv_ab_freeze -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes

# cutoff
# rm -rf results/meta_inh*/*meta_proto* ckpt/meta_inh*/*meta_proto*
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_proto_guide_freeze -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_proto_guide -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes

# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_stop -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_stop -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes

# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_full -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_full -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_freeze -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_freeze -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes

# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_stop -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_stop -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes

# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_full -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_full -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_freeze -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_freeze -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes


# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_proto_stop -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_proto_stop -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes

# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_proto_full -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_proto_full -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_proto_freeze -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_proto_freeze -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes

# rm -rf results/meta_*/CNN_pH_kfold_*_meta_proto_* ckpt/meta_*/CNN_pH_kfold_*_meta_proto_*

# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_proto_stop -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_proto_stop -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes

# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_proto_full -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_proto_full -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_proto_freeze -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_proto_freeze -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes

# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_proto_stop -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_proto_stop -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_proto_full -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_proto_full -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_proto_freeze -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_proto_freeze -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes

# rm -rf results/meta*/*ab* ckpt/meta*/*ab*
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_ab_stop -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_ab_full -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_ab_freeze -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes

# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_ab_stop -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_ab_full -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_ab_freeze -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes

# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_ab2_stop -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_ab2_full -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_ab2_freeze -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes

# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_ab2_stop -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_ab2_full -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_ab2_freeze -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes

# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_ab_stop -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_ab_full -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_ab_freeze -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes

# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_ab_stop -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_ab_full -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_ab_freeze -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes

# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_ab2_stop -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_ab2_full -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_ab2_freeze -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_classes

# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_ab2_stop -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_ab2_full -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_guide_ab2_freeze -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_classes

# rm -rf results/meta*/*meta_multi* ckpt/meta*/*meta_multi*
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_multi_stop -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_chem substrates_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_multi_full -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_chem substrates_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_multi_freeze -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_chem substrates_classes

# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_multi_stop -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_fda inhibitors_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_multi_full -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_fda inhibitors_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_multi_freeze -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_fda inhibitors_classes

# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_multi_stop -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_chem substrates_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_multi_full -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_chem substrates_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_multi_freeze -d results/meta_sub -c1 ckpt/meta_sub -l 1 -t substrates_chem substrates_classes

# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_multi_stop -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_fda inhibitors_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_multi_full -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_fda inhibitors_classes
# python 000_run_multi.py -n CNN_pH_kfold_nval_meta_multi_freeze -d results/meta_inh -c1 ckpt/meta_inh -l 1 -t inhibitors_fda inhibitors_classes
