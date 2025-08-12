import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# let's create a set of locals referring to our directory and working directory 
home_dir = Path.home()
work_dir = (home_dir / 'SIEPR-HOUSING-POLICY')
data = (work_dir / 'data')
raw_data = (data / 'raw')
llm_regulations = (data / 'llm_regulatory_measurement')
clean_data = (data / 'clean')
code = Path.cwd() 
output = (work_dir / 'output')

# first, let's start with the cbsa-level dataset 
cbsa = pd.read_csv(f'{llm_regulations}/cbsa.csv')

cbsa.columns = cbsa.columns.str.lower() 

cbsa.to_csv(f'{clean_data}/cbsa_llm_regulatory_index.csv')