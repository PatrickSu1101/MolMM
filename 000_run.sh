#!/bin/bash

set -e

rm -rf MolMM/data/split/*/sub*
rm -rf MolMM/data/split/*/inh*
rm -rf results/* ckpt/*

./scripts/001_run_baseline.sh
./scripts/002_run_multitask.sh
./scripts/003_run_meta.sh
./scripts/004_run_ablation.sh

./scripts/001_run_baseline_DNN.sh
./scripts/003_run_meta_DNN.sh