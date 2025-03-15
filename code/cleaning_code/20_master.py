import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# let's create a set of locals referring to our directory and working directory 
home = Path.home()
work_dir = (home / 'housing_project')
data = (work_dir / 'data')
raw_data = (data / 'raw')
raw_election = (raw_data / 'election_data')
clean_data = (data / 'clean')
clean_election = (clean_data / 'election_data')
output = (work_dir / 'output')
code = Path.cwd() 

# pull 2020 wharton lui data (already cleaned on the county-level)
wharton_2020 = pd.read_csv(f'{clean_data}/wharton_land_2020.csv')

# pull election results (also on the county level)
pres_county = pd.read_csv(f'{clean_election}/pres_election_2000_2024_county.csv')
house_county = pd.read_csv(f'{clean_election}/house_2020_2024.csv')

# now start master 
master = pd.merge(wharton_2020, pres_county, on=['state', 'county_name'])
master = pd.merge(master, house_county, on=['state', 'county_name'])

# export county-level data 
master.to_csv(f'{clean_data}/master_county_level.csv', index=False)