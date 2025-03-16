import pandas as pd
from pathlib import Path
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
# let's create a set of locals referring to our directory and working directory 
home = Path.home()
work_dir = (home / 'housing_project')
data = (work_dir / 'data')
raw_data = (data / 'raw')
clean_data = (data / 'clean')
output = (work_dir / 'output')
code = Path.cwd() 

# import R packages
rmarkdown = importr('rmarkdown')
rmd_file = code / '05b_state_trifectas.Rmd'
try:
    robjects.r(f"""
    output_dir <- "{clean_data.as_posix()}"
    rmarkdown::render(
        input = "{rmd_file}",
        output_file = file.path(output_dir, "State_Trifectas_Ballotpedia_Scrape.pdf")
    )
    """)
    print(f"R Markdown rendered successfully. Output saved to: {clean_data}")
except Exception as e:
    print(f"Error rendering R Markdown: {e}")

# pull and make small adjustments to file 
state_trifectas = pd.read_csv(f'{clean_data}/state_trifectas_ballotpedia_scrape.csv')
state_trifectas.drop(columns={'Unnamed: 0'}, inplace=True)
state_trifectas.columns = state_trifectas.columns.str.lower() # make all column titles lowercase 
# make all string columns' vaulues entirely lowercase 
state_trifectas = state_trifectas.applymap(lambda x: x.lower() if isinstance(x, str) else x)

state_trifectas.to_csv(f'{clean_data}/state_trifectas_ballotpedia_scrape.csv', index=False)