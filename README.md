# Datasets.py 
*.h5 data generation code. 

Download SHARADAR_SEP and SHARADAR_SFP separately from Quandl, and merge them together to create `SHARADAR_SEFP.csv`, with date and tickre/symbol as multipleIndex. 
Change parsing dates in `prices` accordingly and change `test_params` to adjust training and testing frames in 252 cycle (252 trading days in a year). 
DATA_STORE path is `sefpassets.h5`, save all predictions separately in a *.h5 file or read from DATA_STORE path. 


# Eon_AI *.py 
Live trading code to put on Interactive Broker. 