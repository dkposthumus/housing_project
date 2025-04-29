import pandas as pd
from pathlib import Path
from fredapi import Fred

# let's create a set of locals referring to our directory and working directory 
home_dir = Path.home()
work_dir = (home_dir / 'housing_project')
data = (work_dir / 'data')
raw_data = (data / 'raw')
clean_data = (data / 'clean')
code = Path.cwd() 
output = (work_dir / 'output')

# pull in 30 year mortgage rates and collapse on year 
mortgage30 = pd.read_csv(f'{raw_data}/mortgage30_us.csv')
mortgage30.columns = mortgage30.columns.str.lower()
mortgage30['observation_date'] = pd.to_datetime(mortgage30['observation_date'])
mortgage30['year'] = mortgage30['observation_date'].dt.year 
mortgage30 = mortgage30.groupby('year')['mortgage30us'].mean().reset_index()

# now pull in census data 
census_data = pd.read_csv(f'{raw_data}/acs_1year_2010_2024.csv')
census_data.columns = census_data.columns.str.lower()

# first, merge mortgage and census data on year 
master = pd.merge(census_data, mortgage30, on=['year'], how='left')

# second, calculate affordability index 
master['monthly_payment'] = (
    master['median_home_value'] * 0.8 * ((master['mortgage30us'] / 12) / (1-(1/(1 + (master['mortgage30us'] / 12))**(360))))
)
master['necessary_income'] = ((master['monthly_payment'] * 12) / master['median_household_income'])*100
master['qualifying_income'] = master['monthly_payment'] * 4 * 12
master['affordability_index'] = (master['median_household_income'] / master['qualifying_income']) * 100

master.to_csv(f'{clean_data}/cbsa_characteristics.csv', index=False)