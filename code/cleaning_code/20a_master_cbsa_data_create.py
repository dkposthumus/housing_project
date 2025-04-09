import pandas as pd
from pathlib import Path

# let's create a set of locals referring to our directory and working directory 
home_dir = Path.home()
work_dir = (home_dir / 'housing_project')
data = (work_dir / 'data')
raw_data = (data / 'raw')
clean_data = (data / 'clean')
crosswalks = (data / 'crosswalks')
code = Path.cwd() 
output = (work_dir / 'output')

##################################################################################################################
# Convert LIHTC data into MSA-level sum
##################################################################################################################
# introduce crosswalk 
crosswalk = pd.read_csv(f'{crosswalks}/tract_cbsa_crosswalk.csv', encoding="latin1")
crosswalk.columns = crosswalk.columns.str.lower()
crosswalk = crosswalk.apply(lambda col: col.str.lower() if col.dtype == "object" or pd.api.types.is_string_dtype(col) else col)
# clean data to match what we have in the LIHTC data 
crosswalk['fips2020'] = (((crosswalk['county'] * 1000000) + (crosswalk['tract'] * 100)).astype(int)).astype(str)
crosswalk = crosswalk[['cbsa23', 'cbsaname23', 'fips2020']]
# introduce cbsa_type var and merge 
cbsa_type = pd.read_csv(f'{crosswalks}/cbsa_type.csv', encoding='latin1')
cbsa_type = cbsa_type[['cbsa23', 'cbsatype23']]

# introduce project-level data from HUD
lihtc = pd.read_csv(f'{clean_data}/2021_lihtc_projects.csv')
lihtc['fips2020'] = lihtc['fips2020'].astype(str)
# drop all leading zeroes 
lihtc['fips2020'] = lihtc['fips2020'].astype(str).str.lstrip("0")
crosswalk['fips2020'] = crosswalk['fips2020'].astype(str).str.lstrip("0")
# merge on fips2020
cbsa_master = pd.merge(lihtc, crosswalk, on=['fips2020'], how='outer')
# now collapse on ['cbsa23', 'cbsaname23'], summing 'li_units'
cbsa_master = cbsa_master.groupby(['cbsa23', 'cbsaname23']).agg({'li_units': 'sum'}).reset_index()
cbsa_master.rename(columns={'li_units': 'li_lihtc_units_2021'}, inplace=True)

##################################################################################################################
# Merge MSA-level permit data on
##################################################################################################################
permit = pd.read_csv(f'{clean_data}/prelim_2023_permitting.csv')

cbsa_master = pd.merge(cbsa_master, permit, on=['cbsa23', 'cbsaname23'], how='outer')

##################################################################################################################
# Save master dataset 
##################################################################################################################
cbsa_master = pd.merge(cbsa_master, cbsa_type, on=['cbsa23'], how='outer')
cbsa_master = cbsa_master.apply(lambda col: col.str.lower() if col.dtype == "object" or pd.api.types.is_string_dtype(col) else col)
cbsa_master.to_csv(f'{clean_data}/msa_lihtc_permits_data.csv', index=False)