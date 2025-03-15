import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import redivis

# let's create a set of locals referring to our directory and working directory 
home_dir = Path.home()
work_dir = (home_dir / 'comparative_poli_econ/final_project')
data = (work_dir / 'data')
raw_data = (data / 'raw')
code = Path.cwd() 
output = (work_dir / 'output')

# set up revidis API 
organization = redivis.organization('SUL')
dataset = organization.dataset('corelogic_loan_level_market_analytics')

master = dataset.query("""
    SELECT * FROM `Events`
    LIMIT 1000
""").to_pandas_dataframe()

