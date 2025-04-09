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

prelim_2024 = pd.read_excel(f'{raw_data}/permits_2024_preliminary.xls', header=7, sheet_name='MSA Units')

prelim_2024.columns = prelim_2024.columns.str.lower()

prelim_2024.rename(
    columns = {
        'cbsa': 'cbsa23',
        'name': 'cbsaname23',
        'metro /micro code': 'metro_micro_code',
        'total': 'new_permits_total',
        '1 unit': 'new_permits_1',
        '2 units': 'new_permits_2',
        '3 and 4 units': 'new_permits_3_and_4',
        '5 units or more': 'new_permits_5_or_more',
    }, inplace=True
)

# now make all string columns lowercase values
prelim_2024 = prelim_2024.apply(lambda col: col.str.lower() if col.dtype == "object" or pd.api.types.is_string_dtype(col) else col)
prelim_2024['cbsaname23'] = prelim_2024['cbsaname23'].str.rstrip()
prelim_2024 = prelim_2024[['cbsa23', 'cbsaname23', 'metro_micro_code', 'new_permits_total']]

# export the data
prelim_2024.to_csv(f'{clean_data}/prelim_2023_permitting.csv', index=False)