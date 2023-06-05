# HOIST

Source code, interactive map for the paper [Evidence-Driven Spatio-Temporal COVID-19 Hospitalization Prediction with Ising Dynamics.](https://www.nature.com/articles/s41467-023-38756-3)

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

## Training HOIST for general prediction tasks

We provide the HOIST training script ```train_hoist.py``` for genearl time series prediction tasks. If your data have ```N``` locations and ```T``` time steps, the requried data are:

```dynamic data```: List of arrays. Each array in the list has shape ```(N, T, dynamic_dims_i)```, which belongs to a dynamic feature category, e.g., hospital resource usage data, vaccination data, claims data, etc. The HOIST model use different categories to model the effect of different fields. It is ok if you only have feature category.

```target```: Array with shape ```(N, T, output_dim)```. Prediction targets.

```static``` (optional): List of arrays. Each array in the list has shape ```(N, T, static_dims_i)```, which belongs to a static data category, e.g., population stats, economic stats, etc. The HOIST model use different categories to calculate the similarities between locations. If you don't need the spatial relationships, you can input ```None```.

```distance``` (optional): Array with shape ```(N, N, distance_dims)```. You can provide multiple distance measures, e.g., geographical distance, mobility distance, etc. Each value ```(i,j)``` in a slice matrix ```(N, N)``` denotes the distance between the i-th location and j-th location.

If neither ```static``` nor ```distance``` data is provided, the spatial module won't be used.

The data path parameters are (all data files are read with ```pickle.load```):

```--train_dynamic```: str, path to the training dynamic data.

```--val_dynamic```: str, path to the validation dynamic data.

```--test_dynamic```: str, path to the test dynamic data.

```--train_y```: str, path to the training y data.

```--val_y```: str, path to the validation y data.

```--test_y```: str, path to the test y data.

```--static``` (optional): str, path to the static data.

```--distance ```(optional): str, path to the distance data.

The model parameters are:

``--use_ising``: (optional): ``true`` or ``false``. Whether to use the Ising loss to regularize the model. We found for some time series prediction tasks (e.g., RSV prediction), setting ising term to false may significantly improve the performance since the Ising dynamics may not applicable.

```--dynamic_dims```: List of integers. Dimensions of each category in the dynamic data. Ensure that ```len(dynamic_dims) == len(dynamic_data)```.

```--static_dims``` (optional): List of integers. Dimensions of each category in the static data. Ensure that ```len(static_dims) == len(static_data)```.

``--distance_dims`` (optional): Integer. Number of distance measures.

``--signs`` (optional): List of +1/-1. The effect of each dynamic feature category to the targets. For example, for hospitalization targets, the vaccinations should have negative effect (-1), while disease stats should have positive effect (+1). If None, the effects won't be differentiated. Ensure that ```len(signs) == len(dynamic_data)```.

```--rnn_dim```: Integer. Dimension of RNN.

More parameters are in the ```train_hoist.py``` file.



Sample usage:

```
python .\train_hoist.py --train_dynamic './tmp/train_dynamic.pkl' --train_y './tmp/train_y.pkl' --val_dynamic './tmp/val_dynamic.pkl' --val_y './tmp/val_y.pkl' --test_dynamic './tmp/test_dynamic.pkl' --test_y './tmp/test_y.pkl' --dynamic_dims 5 4 --distance './tmp/distance.pkl' --distance_dims 2 --signs 1 -1
```

## Training HOIST for COVID-19 hospitalization

We provide two versions of HOIST for hospitalization prediction: with and without claims data. For ```train_hoist_without_claims.ipynb```, users can directly train the HOIST without accessing the claims data. The performances are worse than the version with claims data but it still outperforms baseline models.

The training functions and hyperparameters used in our paper are in the ipynb files. It takes ~20 mintues to train the HOIST without claims data for 1 run on a laptop with CPU.

## Acknowledgement

The interactive map is built based on https://github.com/mwaugh0328/covid-19-map.

## Citation

Please cite our work:
```
@article{gao2023evidence,
  title={Evidence-driven spatiotemporal COVID-19 hospitalization prediction with Ising dynamics},
  author={Gao, Junyi and Heintz, Joerg and Mack, Christina and Glass, Lucas and Cross, Adam and Sun, Jimeng},
  journal={Nature Communications},
  volume={14},
  number={1},
  pages={3093},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```
or
```
Gao, J., Heintz, J., Mack, C., Glass, L., Cross, A., & Sun, J. (2023). Evidence-driven spatiotemporal COVID-19 hospitalization prediction with Ising dynamics. Nature Communications, 14(1), 3093.
```
