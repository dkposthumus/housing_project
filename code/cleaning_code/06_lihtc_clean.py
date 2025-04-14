import pandas as pd
from pathlib import Path

# let's create a set of locals referring to our directory and working directory 
home_dir = Path.home()
work_dir = (home_dir / 'housing_project')
data = (work_dir / 'data')
raw_data = (data / 'raw')
clean_data = (data / 'clean')
code = Path.cwd() 
output = (work_dir / 'output')

df_raw = pd.read_csv(f'{raw_data}/lihtcpub.csv')
df = df_raw.copy()
# restrict dataframe to only the necessary variables 
df = df[['hud_id', 'project', 'proj_cty', 'proj_st', 'proj_zip', 'fips2020',
         'yr_pis', 'n_units', 'li_units', 'credit']]

df = df[df['yr_pis'].isin([2020, 2021, 2022])]
# now only filter according to credit (only 1, corresponding to the 30% credit subsidy)
df = df[df['credit'] == 1]
df.drop(columns = {'credit'}, inplace=True)

df.to_csv(f'{clean_data}/2021_lihtc_projects.csv', index=False)