# Computational Prediction of ABCB1-allocrites Interactions: An Integrated Approach via Confidence-aware Pre-training and Coarse-grained Umbrella Sampling

## MolMM-PMF

MolMM-PMF is an integrated computational framework for predicting substrates and inhibitors (allocrites) of the ABCB1 (P-glycoprotein) transporter. The framework combines meta-learning, multi-task learning, and confidence-aware transfer learning, and incorporates explainable AI (SHAP) and coarse-grained molecular dynamics (umbrella sampling) to provide mechanistic insights into allocrite recognition and inhibition. 

This repository provides: 

* Scripts for dataset curation and preprocessing 

* The MolMM model implementation (benchmark and ablation study models )
* Code for feature embeddings via MolMM model
* Code for SHAP-based molecular feature interpretation (“Smaps” and "tree plots") 