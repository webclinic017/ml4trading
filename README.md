# TwoSigma Live
---------------------------------------------------------------
`Datasets.py` is *.h5 data generation code. 
`Eon_AI_sigma.py` is papar trading code to order on sigma API. 

* Download SHARADAR_SEP and SHARADAR_SFP separately from [Quandl](https://data.nasdaq.com/databases/SFA/usage/export), unzip and merge them together to create `SHARADAR_SEFP.csv`, with date and tickre/symbol as multipleIndex. Store it in a desirable directionary because you need to ensure `Datasets.py` reads the correct path in line 25. 

* DATA_STORE path is `sefpassets.h5`, save all predictions separately in a *.h5 file or read from DATA_STORE path. 

* Environment
  1. python 3.6-3.8
  2. conda/anaconda 
  3. zipline-reloaded, downloading from https://github.com/stefan-jansen/machine-learning-for-trading/tree/main/installation
  4. pandas==1.4.x, updated after `conda env create -f installation/uros/ml4t.yml`, for pyfolio and compatible plotting packages. 

* (**Optional**)Change parsing dates in `prices` accordingly and change `test_params` to adjust training and testing frames in 252 cycle (252 trading days in a year). 

# Eon_AI_sigma.py 
* Change `now` for current or last trading day and `DATA_STORE` for where new predictions *.h5. 
* sigma API may encounter latency issues to return results or not working properly. Ensure what stocks with how much shares are put. 

# Eon_AI.py 
Live trading code to put on Interactive Broker. 
