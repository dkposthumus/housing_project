import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
# let's create a set of locals referring to our directory and working directory 
home = Path.home()
work_dir = (home / 'housing_project')
data = (work_dir / 'data')
raw_data = (data / 'raw')
raw_election = (raw_data / 'election_data')
clean_data = (data / 'clean')
clean_election = (clean_data / 'election_data')
output = (work_dir / 'output')
code = Path.cwd() 

