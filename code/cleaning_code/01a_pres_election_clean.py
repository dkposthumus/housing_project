import pandas as pd
from pathlib import Path
# let's create a set of locals referring to our directory and working directory 
home = Path.home()
work_dir = (home / 'medicaid_project')
data = (work_dir / 'data')
raw_data = (data / 'raw')
clean_data = (data / 'clean')
state_level = (clean_data / 'state_level')
state_election_data = (state_level / 'election_results')
cd_level = (clean_data / 'cd_level')
cd_election_data = (cd_level / 'election_results')
output = (work_dir / 'output')
code = Path.cwd() 

historical_election = pd.read_csv(f'{raw_data}/county_pres_2000_2020.csv')
election_2024 = pd.read_excel(f'{raw_data}/county_pres_2024.xlsx', sheet_name='County')

for df in [historical_election, election_2024]:
    df.columns = df.columns.str.lower()
# now make everything lowercase in both datasets
historical_election = historical_election.applymap(lambda x: x.lower() if isinstance(x, str) else x)
election_2024 = election_2024.applymap(lambda x: x.lower() if isinstance(x, str) else x)
'''
We want to produce 2 datasets:
- CD-level 
- State-level 
Each of these datasets needs to have the following features:
- Cover each presidential election from 2000 to 2024
- Contain the percentage of votes at the geographic level of the dataset achieved by the following data categories:
    - Republican 
    - Democrat 
    - all other (including 3rd party candidates)
This baseline dataset will allow us to do everything else, namely calculate swing from year x to year y -- the direction that this analysis is heading towards
'''

'''
Unfortunately, the two datasets we're working with (one historical [2000-2024] and the other current [2024]) are very different in structure.
Let's just clean them separately.'
'''

######################################################################################################################################
# Historical 
######################################################################################################################################
'''
Here are the relevant columns:
- county
- state
- mode (type of voting that's captured by these numbers) 
- candidate/party
- candidatevotes (number of votes candidate won)
- totalvotes (toal votes for this mode of voting)

# we first want to make a long dataset, rather than a wide one w/the following columns:
- year 
- county
- state 
- party (either democrat, republican, or other)
- % of votes for that candidate
- number of votes for that candidate
- total number of votes in that county in that year
'''

# first, let's check how many counties have mode == 'TOTAL'
# test with 'party' == 'DEMOCRATIC'
dem_historical = historical_election[historical_election['party'] == 'democrat']
total_present = dem_historical.groupby(['county_name', 'state', 'year'])['mode'].apply(lambda x: 'total' in x.unique())
missing_total = total_present[~total_present]
#print('County-Year groups missing "total":')
#print(missing_total.index.tolist())

votes_2020 = historical_election[historical_election['year'] == 2020]
# --- Step 1: Identify counties with an existing 'total' entry ---
existing_totals = votes_2020[votes_2020['mode'] == 'total'].copy()
# Create a set of county/state pairs that have a 'total' row
existing_ids = set(existing_totals[['county_name', 'state']].apply(tuple, axis=1))
# --- Step 2: For counties missing a 'total' entry, compute manual totals ---
# Get all county/state pairs in 2020 and then identify those without a 'total' row
all_ids = set(votes_2020[['county_name', 'state']].apply(tuple, axis=1))
missing_ids = all_ids - existing_ids
# Subset votes_2020 to only those counties missing a 'total'
manual_votes = votes_2020[votes_2020[['county_name', 'state']].apply(tuple, axis=1).isin(missing_ids)]
# (We assume here that manual_votes contains no row with mode=='total'.)
# Compute the "total" votes by first collapsing by mode, then summing across modes.
# For example, if for each mode the 'totalvotes' should be averaged and then added up:
total_manual = (manual_votes
                .groupby(['county_name', 'state', 'mode'])
                .agg({'totalvotes': 'mean'})
                .reset_index()
                .groupby(['county_name', 'state'])
                .agg({'totalvotes': 'sum'})
                .reset_index())
# Aggregate candidate votes by summing by party
candidates_manual = (manual_votes
                     .groupby(['county_name', 'state', 'party'])
                     .agg({'candidatevotes': 'sum'})
                     .reset_index())
# Merge the total and candidate-level aggregates for manual counties
manual_totals = pd.merge(total_manual, candidates_manual, on=['county_name', 'state'], how='outer')
manual_totals['year'] = 2020
# --- Step 3: Combine the results ---
# Here, final_votes_2020 includes both counties with existing totals and those computed manually
final_votes_2020 = pd.concat([existing_totals, manual_totals], axis=0, ignore_index=True)
# --- Step 4: Replace the 2020 data in the full historical_election DataFrame ---
# Remove all 2020 rows from the original DataFrame (or those not needed) and append our final 2020 data.
historical_election_no2020 = historical_election[historical_election['year'] != 2020]
final_historical_election = pd.concat([historical_election_no2020, final_votes_2020], 
                                      axis=0, ignore_index=True)
# now reshape wide for candidatevotes BY party
pivoted_table = historical_election.pivot_table(
    index=['year', 'state', 'county_name'], 
    columns='party', 
    values='candidatevotes', 
    aggfunc='sum', 
    fill_value=0  # fills missing values with 0
).reset_index()
pivoted_table['other'] = pivoted_table['green'] + pivoted_table['libertarian'] + pivoted_table['other']
pivoted_table.drop(columns={'green', 'libertarian'}, inplace=True)
# (Assuming totalvotes is the same across rows for a given county-year, you can take the max or first.)
total_table = historical_election.groupby(['year', 'state', 'county_name'])['totalvotes'].max().reset_index()
# Step 4: Merge the pivoted candidate votes with totalvotes.
historical_election = pd.merge(pivoted_table, total_table, on=['year', 'state', 'county_name'])
# Step 5: Rename the columns to match your desired names.
historical_election.rename(columns={
    'democrat': 'democratic_pres_votes',
    'republican': 'republican_pres_votes',
    'other': 'other_pres_votes',
    'totalvotes': 'total_pres_votes'
}, inplace=True)

######################################################################################################################################
# 2024 
######################################################################################################################################
# now filter 2024 election data to include only Trump
election_2024 = election_2024[['state', 'county_name', 'trump', 'harris',
                            'total vote', 'lsad_trans']]
election_2024 = election_2024[election_2024['lsad_trans'].isin(['county', 'parish'])]
# we have to miscellaneously do some renaming 
for problem, solution in zip(['lasalle', 'st. louis', 'coös'],
                            ['la salle', 'st. louis county', 'coos']):
    election_2024.loc[election_2024['county_name'] == problem, 'county_name'] = solution
election_2024.loc[election_2024['state'] == 'dc', 'state'] = 'district of columbia'
election_2024 = election_2024.dropna(subset=['county_name']) # drop all rows where 'county_name' is missing
# now rename all states, which are abbreviations to the lowercase state names
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
state_mapping = {key.lower(): value for key, value in state_mapping.items()}
election_2024['state'] = election_2024['state'].map(lambda x: state_mapping.get(x, x.lower()))
election_2024['year'] = 2024 
election_2024['other_pres_votes'] = election_2024['total vote'] - (election_2024['trump'] - election_2024['harris'])
election_2024.rename(
    columns = {
        'harris': 'democratic_pres_votes',
        'trump': 'republican_pres_votes',
        'total vote': 'total_pres_votes'
    }, inplace=True
)
election_2024.drop(columns={'lsad_trans'}, inplace=True)

######################################################################################################################################
# merge and clean datasets
######################################################################################################################################
master = pd.concat([historical_election, election_2024], axis=0, ignore_index=True)
master.to_csv(f'{clean_data}/pres_election_2000_2024_county.csv', index=False)

# now we want to produce a state-level total vote dataset
# this is very easy; we can just collapse on state
master.drop(columns={'county_name'}, inplace=True)
master = master.groupby(['year', 'state']).agg({
    'democratic_pres_votes': 'sum',
    'republican_pres_votes': 'sum',
    'other_pres_votes': 'sum',
    'total_pres_votes': 'sum'
}).reset_index()
master.to_csv(f'{state_election_data}/pres_election_2000_2024_state.csv', index=False)

# now we want CD-level data, which is a little more complex 
