import pandas as pd
from pathlib import Path
import requests
import time

# let's create a set of locals referring to our directory and working directory 
home_dir = Path.home()
work_dir = (home_dir / 'housing_project')
data = (work_dir / 'data')
raw_data = (data / 'raw')
clean_data = (data / 'clean')
code = Path.cwd() 
output = (work_dir / 'output')

CENSUS_API_KEY = '680d2ddd22bf2e45296e8e9e8040df38e7740f7d'
DATASET = 'acs/acs1'  # 1-year ACS data
variables = {
    'B01003_001E': 'Total_Population',
    'B19013_001E': 'Median_Household_Income',
    'B25064_001E': 'Median_Gross_Rent',
    'B25077_001E': 'Median_Home_Value',
    'B25003_001E': 'Total_Housing_Units',
    'B25003_002E': 'Owner_Occupied_Housing_Units'
}
# 📌 List of years (ACS 1-year)
years = [2010, 2024]
# 📌 Base URL pattern
BASE_URL_PATTERN = 'https://api.census.gov/data/{year}/acs/acs1'
# 📌 Collect all years into a list
all_years_data = []
for year in years:
    print(f"Fetching ACS 1-year data for {year}...")
    base_url = BASE_URL_PATTERN.format(year=year)
    params = {
        'get': ','.join(variables.keys()),
        'for': 'metropolitan statistical area/micropolitan statistical area:*',
        'key': CENSUS_API_KEY
    }
    response = requests.get(base_url, params=params)
    
    if response.status_code != 200:
        print(f"Warning: Failed to fetch data for {year}. Skipping...")
        continue
    
    data = response.json()
    columns = data[0]
    rows = data[1:]
    
    df = pd.DataFrame(rows, columns=columns)
    
    # Convert numeric columns
    for var in variables.keys():
        df[var] = pd.to_numeric(df[var], errors='coerce')
    
    # Rename columns
    df = df.rename(columns=variables)
    
    df['Year'] = year  # 📌 add year info
    
    all_years_data.append(df)
    
    time.sleep(0.5)  # Be nice to the API!

# 📌 Combine into a single dataframe
final_df = pd.concat(all_years_data, ignore_index=True)

# 📌 Preview
print(final_df.head())

# 📌 Optional: save to CSV
final_df.rename(
    columns = {
        'metropolitan statistical area/micropolitan statistical area': 'cbsa'
    }, inplace=True
)
final_df.to_csv(f'{raw_data}/acs_1year_2010_2024.csv', index=False)