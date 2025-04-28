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

# pull wharton_2020 data
wharton_2020 = pd.read_stata(f'{raw_data}/WRLURI_01_15_2020.dta')
# let's rename key variables 
wharton_2020.columns = wharton_2020.columns.str.lower()
wharton_2020.rename(
    columns = {
        'lppi18': 'local_political_pressure_2018',
        'spii18': 'state_involvement_2018',
        'cii18': 'court_involvement_2018',
        'lpai18': 'local_project_2018',
        'lzai18': 'local_zoning_2018',
        'lai18': 'local_assembly_2018',
        'sri18': 'supply_restrictions_2018',
        'dri18': 'density_restriction_2018',
        'osi18': 'open_space_2018',
        'ei18': 'exactions_2018',
        'ahi18': 'affordable_housing_2018',
        'adi18': 'approval_delay_2018'
    }, inplace=True
)
wharton_2020['year'] = 2018
state_mapping = {
    'AL': 'alabama', 'AK': 'alaska', 'AZ': 'arizona', 
    'AR': 'arkansas', 'CA': 'california',
    'CO': 'colorado', 'CT': 'connecticut', 'DE': 'delaware', 
    'FL': 'florida', 'GA': 'georgia',
    'HI': 'hawaii', 'ID': 'idaho', 'IL': 'illinois', 
    'IN': 'indiana', 'IA': 'iowa',
    'KS': 'kansas', 'KY': 'kentucky', 'LA': 'louisiana', 
    'ME': 'maine', 'MD': 'maryland',
    'MA': 'massachusetts', 'MI': 'michigan', 'MN': 'minnesota', 
    'MS': 'mississippi', 'MO': 'missouri',
    'MT': 'montana', 'NE': 'nebraska', 'NV': 'nevada', 
    'NH': 'new hampshire', 'NJ': 'new jersey',
    'NM': 'new mexico', 'NY': 'new york', 
    'NC': 'north carolina', 'ND': 'north dakota',
    'OH': 'ohio', 'OK': 'oklahoma', 'OR': 'oregon', 
    'PA': 'pennsylvania', 'RI': 'rhode island',
    'SC': 'south carolina', 'SD': 'south dakota', 'TN': 'tennessee', 
    'TX': 'texas', 'UT': 'utah',
    'VT': 'vermont', 'VA': 'virginia', 'WA': 'washington', 
    'WV': 'west virginia', 'WI': 'wisconsin', 'WY': 'wyoming'
}
wharton_2020['state'] = wharton_2020['state'].str.strip().str.upper()
wharton_2020['state'] = wharton_2020['state'].map(state_mapping)
# replace all string values in dataset with lowercase 
wharton_2020 = wharton_2020.applymap(lambda x: x.lower() if isinstance(x, str) else x)
wharton_2020['county_name'] = wharton_2020['countyname18'].str.replace(" county", "", regex=False)

# export
wharton_2020.to_csv(f'{clean_data}/wharton_land_2020.csv', index=False)

# pull wharton_2008 data
wharton_2008 = pd.read_stata(f'{raw_data}/WHARTON LAND REGULATION DATA_1_24_2008.dta')
wharton_2008.columns = wharton_2008.columns.str.lower()
wharton_2008['year'] = 2008

# export
wharton_2008.to_csv(f'{clean_data}/wharton_land_2008.csv', index=False)