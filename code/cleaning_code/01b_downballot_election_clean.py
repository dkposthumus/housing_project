import pandas as pd
from pathlib import Path
import warnings # suppress noisy warnings in jupyter output
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

# let's create a set of locals referring to our directory and working directory 
home = Path.home()
work_dir = (home / 'medicaid_project')
data = (work_dir / 'data')
raw_data = (data / 'raw')
clean_data = (data / 'clean')
state_level = (clean_data / 'state_level')
output = (work_dir / 'output')
code = Path.cwd() 

# define function that automatically and quite nicely cleans 2024 election data for us
def data_2024_cleaning(df, office):
    df.rename(columns={'Unnamed: 0': 'county_name', 'Unnamed: 1': 'state'}, inplace=True)
    df.dropna(subset=['county_name'], inplace=True)
    df.dropna(subset=['state'], inplace=True)
    # rename columns to be more descriptive:
    df.rename(
        columns= {'Total Vote': f'{office} totalvotes, 2024',
                  'Democratic.1': f'{office} dem votecount, 2024',
                  'Republican.1': f'{office} rep votecount, 2024',},
                  inplace=True
    )
    # now keep only observations for which LSAD
    df = df[df['LSAD_TRANS'].isin(['County', 'Parish'])]
    # exclude rows referencing state totals
    df = df[df['state'] != 'T']
    cols_interest = ['county_name', 'state', f'{office} totalvotes, 2024', 
                     f'{office} dem votecount, 2024', f'{office} rep votecount, 2024']
    df = df[cols_interest]
    # now make all variable values lowercase
    df = df.apply(lambda x: x.astype(str).str.lower())
    return df

####################################################################################
# let's start with cleaning the house election data 
house_2024 = pd.read_excel(f'{raw_data}/county_house_2024.xlsx', sheet_name='County')
house_2024 = data_2024_cleaning(house_2024, 'house')

####################################################################################
# next, let's clean the senate election data
senate_2024 = pd.read_excel(f'{raw_data}/county_senate_2024.xlsx', sheet_name='County')
senate_2024 = data_2024_cleaning(senate_2024, 'senate')

####################################################################################
# next, let's extract / clean the 2020 data
def clean_2020_data(states, full_states, office):
    states_data = {}
    for state, full_name in zip(states, full_states):
        print(f'Cleaning {state} {office} data')
        df = pd.read_csv(f'{raw_data}/2020_precinct_state/2020-{state}-precinct-general.csv')
        # make all variable values lowercase
        df = df.apply(lambda x: x.astype(str).str.lower())
        df['votes'] = pd.to_numeric(df['votes'], errors='coerce')
        # now only keep where dataverse == office
        df = df[df['dataverse'] == office]
        # first, if there is no county name, skip data set
        if len(df['county_name'].unique()) != 0:
            # now sum those by county 
            df[f'{office} totalvotes, 2020'] = df.groupby('county_name')['votes'].transform('sum')
            collapsed_df = df.groupby('county_name')[f'{office} totalvotes, 2020'].mean().reset_index()
            # now calculate the vote share for each party
            party_data = {}
            for party, label in zip(['democrat', 'republican'], ['dem', 'rep']):
                party_df = df[df['party_simplified'] == party]
                party_df[f'{office} {label} votecount, 2020'] = party_df.groupby('county_name')['votes'].transform('sum')
                party_df = party_df.groupby('county_name')[f'{office} {label} votecount, 2020'].mean().reset_index()
                party_data[label] = party_df
            df = pd.merge(collapsed_df, party_data['dem'], on='county_name', how='outer')
            df = pd.merge(df, party_data['rep'], on='county_name', how='outer')
            for label in ['dem', 'rep']:
                df[f'{label}_pct_{office}_2020'] = df[f'{office} {label} votecount, 2020'] / df[f'{office} totalvotes, 2020']
            df['state'] = full_name
            states_data[state] = df 
        else:
            print(f'{state} has no county-level data')
    # now concatenate all the dataframes
    all_states = pd.concat(states_data.values())
    return all_states

# create list of states that consist of lowercase abbreviation
states = ['al', 'ak', 'az', 'ar', 'ca', 'co', 'ct', 'de', 'fl', 'ga', 'hi', 
          'id', 'il', 'in', 'ia', 'ks', 'ky', 'la', 'me', 'md', 'ma', 'mi', 'mn', 
          'ms', 'mo', 'mt', 'ne', 'nv', 'nh', 'nj', 'nm', 'ny', 'nc', 'nd', 'oh', 'ok', 
          'or', 'pa', 'ri', 'sc', 'sd', 'tn', 'tx', 'ut', 'vt', 'va', 'wa', 'wv', 'wi', 'wy']
# create list of full state names, lOWERCASE
state_full = [
    'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado', 'connecticut', 'delaware',
    'florida', 'georgia', 'hawaii', 'idaho', 'illinois', 'indiana', 'iowa', 'kansas', 'kentucky',
    'louisiana', 'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota', 'mississippi', 
    'missouri', 'montana', 'nebraska', 'nevada', 'new hampshire', 'new jersey', 'new mexico', 
    'new york', 'north carolina', 'north dakota', 'ohio', 'oklahoma', 'oregon', 'pennsylvania', 
    'rhode island', 'south carolina', 'south dakota', 'tennessee', 'texas', 'utah', 'vermont', 
    'virginia', 'washington', 'west virginia', 'wisconsin', 'wyoming'
]

state_map = dict(zip(states, state_full))
for df in [house_2024, senate_2024]:
    df['state'] = df['state'].map(lambda x: state_map.get(x, x))

house_2020 = clean_2020_data(states, state_full, 'house')
house_2020_2024 = pd.merge(house_2020, house_2024, on=['county_name', 'state'], how='outer')
house_2020_2024.to_csv(f'{clean_data}/house_2020_2024.csv', index=False)

senate_2020 = clean_2020_data(states, state_full, 'senate')
senate_2020_2024 = pd.merge(senate_2020, senate_2024, on=['county_name', 'state'], how='outer')
senate_2020_2024.to_csv(f'{clean_data}/senate_2020_2024.csv', index=False)

####################################################################################
# next, let's extract / clean the 2020 data