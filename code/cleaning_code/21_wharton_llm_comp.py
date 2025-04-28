import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# let's create a set of locals referring to our directory and working directory 
home_dir = Path.home()
work_dir = (home_dir / 'housing_project')
data = (work_dir / 'data')
raw_data = (data / 'raw')
clean_data = (data / 'clean')
cleaning_code = Path.cwd() 
output = (work_dir / 'output')

# now, pull county-level wharton land use index
wharton = pd.read_csv(f'{clean_data}/wharton_land_2020.csv')


# first, pull cbsa-level regulatory dataset 
llm_regulations = pd.read_csv(f'{clean_data}/cbsa_llm_regulatory_index.csv')

