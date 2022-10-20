# HOIST

Source code, interactive map for the paper HOIST: Evidence-Driven Spatio-Temporal COVID-19 Hospitalization Prediction with Ising Dynamics.

Please visit https://v1xerunt.github.io/HOIST/ for the interactive result map.

## Requirements

* Install python, pytorch. We use Python 3.7.6, Pytorch 1.12.1.
* If you plan to use GPU computation, install CUDA

## Data resources

- The mobility data are collected from https://github.com/GeoDS/COVID19USFlows.
- The county level data are collected from https://github.com/JieYingWu/COVID-19_US_County-level_Summaries.
- The infected case numbers are collected from https://github.com/CSSEGISandData/COVID-19.
- The medical resource data are collected from https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/g62h-syeh.
- The claims and vaccination data are collected from https://www.iqvia.com/solutions/real-world-evidence.
- We provide the processed version of all publicly available data in the data folder.

## Training HOIST

We provide two versions of HOIST: with and without claims data. For ```train_hoist_without_claims.ipynb```, users can directly train the HOIST without accessing the claims data. The performances are worse than the version with claims data but it still outperforms baseline models.

The training functions and hyperparameters used in our paper are in the ipynb files. It takes ~20 mintues to train the HOIST without claims data for 1 run on a laptop with CPU.

## Acknowledgement

The interactive map is built based on https://github.com/mwaugh0328/covid-19-map.