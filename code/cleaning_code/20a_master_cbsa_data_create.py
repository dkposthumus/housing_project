import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

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
cbsa_master['li_units'] = cbsa_master['li_units'].fillna(0)
cbsa_master.fillna(0, inplace=True)
# now collapse on ['cbsa23', 'cbsaname23'], summing 'li_units'
cbsa_master = cbsa_master.groupby(['cbsa23', 'cbsaname23', 'yr_pis']).agg({'li_units': 'sum'}).reset_index()
cbsa_master.rename(columns={'li_units': 'li_lihtc_units'}, inplace=True)
for year in [2020, 2021, 2022]:
    year_df = cbsa_master[(cbsa_master['yr_pis'] == year) | (cbsa_master['yr_pis'] == 0)]
    zero_share_avg = (year_df['li_lihtc_units'] == 0).mean()
    print(f"{zero_share_avg:.2%} of CBSAs have 0 average LIHTC units in {year}")
    sns.kdeplot(data=year_df, x='li_lihtc_units', fill=False, label=year)
cbsa_master = cbsa_master.groupby(['cbsa23', 'cbsaname23']).agg({'li_lihtc_units': 'mean'}).reset_index()
sns.kdeplot(data=year_df, x='li_lihtc_units', fill=False, label='2020-2022 3-Year Average')
zero_share_avg = (cbsa_master['li_lihtc_units'] == 0).mean()
print(f"3-Year Average: {zero_share_avg:.2%} of CBSAs have 0 average LIHTC units")
plt.title("KDE of Low-Income LIHTC Units")
plt.xlabel("Volume")
plt.ylabel("Density")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

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