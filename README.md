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

## Training HOIST

The training functions and hyperparameters are in the ```train_hoist.py```. 

## Acknowledgement

The interactive map is built based on https://github.com/mwaugh0328/covid-19-map.