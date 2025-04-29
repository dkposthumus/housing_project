import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import geopandas as gpd

# let's create a set of locals referring to our directory and working directory 
home_dir = Path.home()
work_dir = (home_dir / 'housing_project')
data = (work_dir / 'data')
raw_data = (data / 'raw')
clean_data = (data / 'clean')
output = (work_dir / 'output')
shapefiles = (data / 'shapefiles')

# now, pull county-level wharton land use index
wharton = pd.read_csv(f'{clean_data}/wharton_land_2020.csv')
vars_to_aggregate = [
    'local_political_pressure_2018', 'state_involvement_2018',
    'court_involvement_2018', 'local_project_2018',
    'local_zoning_2018', 'local_assembly_2018',
    'supply_restrictions_2018', 'density_restriction_2018',
    'open_space_2018', 'exactions_2018',
    'affordable_housing_2018', 'approval_delay_2018',
    'wrluri18'
]
# Step 2: Subset and drop missing weight_cbsa or cbsacode18
df_cbsa = wharton[['cbsa', 'weight_cbsa'] + vars_to_aggregate].dropna(subset=['cbsa', 'weight_cbsa'])
# Step 3: For safety, also drop rows where all vars_to_aggregate are missing
df_cbsa = df_cbsa.dropna(subset=vars_to_aggregate, how='all')
# Step 4: Define a function to compute weighted averages
def weighted_avg(group):
    w = group['weight_cbsa']
    return (group[vars_to_aggregate].multiply(w, axis=0)).sum() / w.sum()
# Step 5: Group by CBSA and apply
cbsa_aggregated = df_cbsa.groupby('cbsa').apply(weighted_avg).reset_index()

# next, pull cbsa-level regulatory dataset 
llm_regulations = pd.read_csv(f'{clean_data}/cbsa_llm_regulatory_index.csv')
# now merge on cbsa 
master = pd.merge(llm_regulations, cbsa_aggregated, on=['cbsa'], how='outer', indicator=True)
master.to_csv(f'{clean_data}/cbsa_regulation_data.csv', index=False)

############################################################################################################
# Check Data Coverage
############################################################################################################
# let's do a simple kde plot comparing the distribution of cbsa's that are missing from the LLM data 
master['z_score'] = (master['overall_index'] - master['overall_index'].mean()) / master['overall_index'].std()
llm_both = master.loc[master['_merge'] == 'both', 'z_score'].dropna()
llm_right_only = master.loc[master['_merge'] == 'left_only', 'z_score'].dropna()

# Step 2: Create KDE objects
kde_both = gaussian_kde(llm_both)
kde_right_only = gaussian_kde(llm_right_only)

# Step 3: Create a common x-axis grid
xmin = min(llm_both.min(), llm_right_only.min())
xmax = max(llm_both.max(), llm_right_only.max())
xgrid = np.linspace(xmin, xmax, 500)

# Step 4: Plot
plt.figure(figsize=(10,6))
plt.plot(xgrid, kde_both(xgrid), label='Present in Both Wharton and Generative Datasets', color='blue')
plt.plot(xgrid, kde_right_only(xgrid), label='Only Present in Generative Regulatory Measure', color='orange')

plt.title('KDE of Generative Regulatory Overall Index Values')
plt.xlabel('Regulatory Overall Index Value')
plt.ylabel('Density')
plt.legend()
plt.show()

# now let's make a heatmap of all cbsa's in the country
cbsa_shapes = gpd.read_file(f'{shapefiles}/tl_2024_us_cbsa/tl_2024_us_cbsa.shp')  # e.g., 'cbsa_shapefile/tl_2020_us_cbsa.shp'

# Step 2: Make sure the CBSA identifier matches
# Check what the CBSA code column is called. Often it is 'CBSAFP' in Census shapefiles.
print(cbsa_shapes.columns)

# Step 3: Merge your master dataset with the shapefile
# Suppose your shapefile has 'CBSAFP' as CBSA code
cbsa_shapes['CBSAFP'] = cbsa_shapes['CBSAFP'].astype(str).str.zfill(5)  # zero-pad if needed
cbsa_shapes = cbsa_shapes[~cbsa_shapes['NAME'].str.contains('AK', na=False)]
cbsa_shapes = cbsa_shapes[~cbsa_shapes['NAME'].str.contains('HI', na=False)]

master['cbsa'] = master['cbsa'].astype(int)
master['cbsa'] = master['cbsa'].astype(str).str.zfill(5)     # match formatting

cbsa_map = cbsa_shapes.merge(master, left_on='CBSAFP', right_on='cbsa', how='left')

# Step 4: Plot the heatmap
fig, ax = plt.subplots(1, 1, figsize=(14, 10))

cbsa_map.plot(
    column='z_score',        # or 'overall_index' or any other variable you want
    cmap='coolwarm',          # color map; alternatives: 'viridis', 'plasma', 'RdYlBu'
    linewidth=0.1,
    ax=ax,
    edgecolor='black',
    legend=True,
    legend_kwds={'label': "Generative Regulatory Index (z-score)", 'shrink': 0.5}
)

# Step 5: Adjust plot
ax.set_title('CBSA-Level Heatmap of Generative Regulatory Index', fontsize=16)
ax.axis('off')

plt.show()

############################################################################################################
# Scatter of normalized overall index 
############################################################################################################
master = master[master['_merge'] == 'both']
master.drop(columns={'_merge'}, inplace=True)

for llm_var, var_label in zip(['overall_index', 'first_pc',
                               'second_pc'],
                               ['Overall', 'Value Capture',
                                'Exclusionary Zoning']):
    # now let's create standardized (i.e. z-scores for the two overall indices)
    for var in [llm_var, 'wrluri18']:
        master[f'{var}_z'] = (master[var] - master[var].mean()) / master[var].std()
    # now let's plot a scatterplot of the two 
    x = master[f'{llm_var}_z']
    y = master['wrluri18_z']
    cbsa_names = master['cbsa_name']  # Assuming you have this column

    plt.scatter(x, y)
    m, b = np.polyfit(x, y, 1)  # 1 = degree of polynomial -> linear fit
    plt.plot(x, m*x + b, color='red', label=f'Best Fit: y = {m:.2f}x + {b:.2f}', linestyle='--')
    '''for xi, yi, label in zip(x, y, cbsa_names):
        plt.text(xi, yi, label, fontsize=6, ha='right', va='bottom')'''

    plt.axvline(x=0)
    plt.axhline(y=0)

    plt.title('Scatterplot of Normalized Housing Regulatory Indices')
    plt.xlabel(f'LLM {var_label} Index (Z-Score)')
    plt.ylabel('Wharton Land Use Survey (2018) (Z-Score)')
    plt.legend()
    plt.tight_layout()
    plt.show()

############################################################################################################
# Scatter of regulations and housing affordability
############################################################################################################
cbsa_characteristics = pd.read_csv(f'{clean_data}/cbsa_characteristics.csv')
cbsa_characteristics['cbsa'] = cbsa_characteristics['cbsa'].astype(str)

# now keep only the affordability index and year 
cbsa_characteristics = cbsa_characteristics[['year', 'cbsa', 'affordability_index']]
# now reshape wide so we have columns, corersponding to each year's affordability index
cbsa_wide = cbsa_characteristics.pivot(index='cbsa', columns='year', values='affordability_index').reset_index()
cbsa_wide.rename(
    columns = {
        2010: 'affordability_index_2010',
        2023: 'affordability_index_2023'
    }, inplace=True
)
master = pd.merge(master, cbsa_wide, on=['cbsa'], how='left')

master.to_csv(f'{clean_data}/check.csv')

for var in ['overall_index', 'affordability_index_2023', 'wrluri18']:
    master[f'{var}_z'] = (master[var] - master[var].mean()) / master[var].std()

temp = master.dropna(subset=['overall_index_z', 'affordability_index_2023_z'])
x = temp['overall_index_z']
y = temp['affordability_index_2023_z']
cbsa_names = temp['cbsa_name']  # Assuming you have this column

plt.scatter(x, y)
m, b = np.polyfit(x, y, 1)  # 1 = degree of polynomial -> linear fit
plt.plot(x, m*x + b, color='red', label=f'Best Fit: y = {m:.2f}x + {b:.2f}', linestyle='--')
for xi, yi, label in zip(x, y, cbsa_names):
        plt.text(xi, yi, label, fontsize=6, ha='right', va='bottom')

plt.axvline(x=0)
plt.axhline(y=0)

plt.title('Scatterplot of Normalized Housing Affordability and LLM Regulation Index')
plt.xlabel('LLM Regulation Index (Z-Score)')
plt.ylabel('Housing Affordability Index (2023) (Z-Score)')
plt.legend()
plt.tight_layout()
plt.show()

# now let's look at the 2010-2023 CHANGE (%) in affordability
temp['affordability_pct_chg'] = (temp['affordability_index_2023'] 
                               - temp['affordability_index_2010']) / temp['affordability_index_2010']
temp['affordability_pct_chg_z'] = (temp['affordability_pct_chg'] 
                                   - temp['affordability_pct_chg'].mean()) / temp['affordability_pct_chg'].std()
temp = temp.dropna(subset=['affordability_pct_chg_z'])
x = temp['overall_index_z']
y = temp['affordability_pct_chg_z']
cbsa_names = temp['cbsa_name']  # Assuming you have this column

plt.scatter(x, y)
m, b = np.polyfit(x, y, 1)  # 1 = degree of polynomial -> linear fit
plt.plot(x, m*x + b, color='red', label=f'Best Fit: y = {m:.2f}x + {b:.2f}', linestyle='--')
for xi, yi, label in zip(x, y, cbsa_names):
        plt.text(xi, yi, label, fontsize=6, ha='right', va='bottom')

plt.axvline(x=0)
plt.axhline(y=0)

plt.title('Scatterplot of Normalized Housing Affordability % Change and LLM Regulation Index')
plt.xlabel('LLM Regulation Index (Z-Score)')
plt.ylabel('Housing Affordability Index % Change (2010 - 2023) (Z-Score)')
plt.legend()
plt.tight_layout()
plt.show()

temp = master.dropna(subset=['wrluri18_z', 'affordability_index_2023_z'])
x = temp['wrluri18_z']
y = temp['affordability_index_2023_z']
cbsa_names = temp['cbsa_name']  # Assuming you have this column

plt.scatter(x, y)
m, b = np.polyfit(x, y, 1)  # 1 = degree of polynomial -> linear fit
plt.plot(x, m*x + b, color='red', label=f'Best Fit: y = {m:.2f}x + {b:.2f}', linestyle='--')
for xi, yi, label in zip(x, y, cbsa_names):
        plt.text(xi, yi, label, fontsize=6, ha='right', va='bottom')

plt.axvline(x=0)
plt.axhline(y=0)

plt.title('Scatterplot of Normalized Housing Affordability and Wharton Regulation Index')
plt.xlabel('Wharton Regulation Index (Z-Score)')
plt.ylabel('Housing Affordability Index (2023) (Z-Score)')
plt.legend()
plt.tight_layout()
plt.show()

############################################################################################################
# Scatter of regulations/affordability and permits 
############################################################################################################
permit = pd.read_csv(f'{clean_data}/prelim_2023_permitting.csv')
permit.rename(
    columns = {
          'cbsa23': 'cbsa'
    }, inplace=True
)
permit['cbsa'] = permit['cbsa'].astype(str)
permit = permit[['cbsa', 'new_permits_total']]
master = pd.merge(master, permit, on=['cbsa'], how='left')

cbsa_characteristics = pd.read_csv(f'{clean_data}/cbsa_characteristics.csv')
cbsa_characteristics['cbsa'] = cbsa_characteristics['cbsa'].astype(str)
cbsa_characteristics = cbsa_characteristics[cbsa_characteristics['year'] == 2023]
# now keep only the affordability index and year 
cbsa_characteristics = cbsa_characteristics[['cbsa', 'total_housing_units']]
master = pd.merge(master, cbsa_characteristics, on=['cbsa'], how='left')

# now construct permit as share of housing stock variable 
master['permits_share'] = master['new_permits_total'] / master['total_housing_units']

for var in ['overall_index', 'affordability_index_2023', 'permits_share']:
    master[f'{var}_z'] = (master[var] - master[var].mean()) / master[var].std()

temp = master.dropna(subset=['overall_index_z', 'affordability_index_2023_z', 'permits_share_z'])
x = temp['overall_index_z']
y = temp['permits_share_z']
cbsa_names = temp['cbsa_name']  # Assuming you have this column

plt.scatter(x, y)
m, b = np.polyfit(x, y, 1)  # 1 = degree of polynomial -> linear fit
plt.plot(x, m*x + b, color='red', label=f'Best Fit: y = {m:.2f}x + {b:.2f}', linestyle='--')
for xi, yi, label in zip(x, y, cbsa_names):
        plt.text(xi, yi, label, fontsize=6, ha='right', va='bottom')

plt.axvline(x=0)
plt.axhline(y=0)

plt.title('Scatterplot of Normalized New Permit Issuance and LLM Regulation Index')
plt.xlabel('LLM Regulation Index (Z-Score)')
plt.ylabel('New Permits (Share of Housing Stock) (2024) (Z-Score)')
plt.legend()
plt.tight_layout()
plt.show()

x = temp['permits_share_z']
y = temp['affordability_index_2023_z']

plt.scatter(x, y)
m, b = np.polyfit(x, y, 1)  # 1 = degree of polynomial -> linear fit
plt.plot(x, m*x + b, color='red', label=f'Best Fit: y = {m:.2f}x + {b:.2f}', linestyle='--')
for xi, yi, label in zip(x, y, cbsa_names):
        plt.text(xi, yi, label, fontsize=6, ha='right', va='bottom')

plt.axvline(x=0)
plt.axhline(y=0)

plt.title('Scatterplot of Normalized New Permit Issuance and Housing Affordability')
plt.xlabel('New Permits (Share of Housing Stock) (2024) (Z-Score)')
plt.ylabel('Housing Affordability (Z-Score)')
plt.legend()
plt.tight_layout()
plt.show()