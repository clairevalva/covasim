'''
Set the parameters for COVID-ABM, pulled from the cruise ship part
'''

import os
import pylab as pl
import pandas as pd
from datetime import datetime


__all__ = ['make_pars', 'load_data']


def make_pars():
    ''' Set parameters for the simulation '''
    pars = {}

    # Simulation parameters
    pars['n_ppl']   = int(33.5*(10**3)) #number of people
    pars['day_0']      = datetime(2020, 1, 22) # Start day of the epidemic
    pars['n_days']     = 70 # How many days to simulate -- 31 days is until 2020-Feb-20
    pars['rand_seed']  = 1 # Random seed, if None, don't reset
    pars['verbose']    = 1 # Whether or not to display information during the run -- options are 0 (silent), 1 (default), 2 (everything)

    # Epidemic parameters
    pars['r_contact']      = .3 # Probability of infection per contact, estimated - beta??
    pars['incub']          = 14 # Using Mike's Snohomish number 
    pars['incub_std']      = 0 # Standard deviation of the serial interval, estimated
    pars['dur']            = 10 # Using Mike's Snohomish number
    pars['dur_std']        = 0 # Variance in duration, set to 0 for replicates
    pars['sensitivity']    = 1.0 # Probability of a true positive, estimated
    pars['symptomatic']    = 5 # Increased probability of testing someone symptomatic, estimated

   

    return pars


# deleted the load_data function? 

def load_data(filename=None):
    ''' Load data for comparing to the model output '''

    default_datafile = 'reported_infections.xlsx'

    # Handle default filename
    if filename is None:
        cwd = os.path.abspath(os.path.dirname(__file__))
        filename = os.path.join(cwd, default_datafile)

    # Load data
    raw_data = pd.read_excel(filename)

    # Confirm data integrity and simplify
    cols = ['day', 'date', 'new_tests', 'new_positives', 'confirmed_crew', 'confirmed_guests', 'evacuated', 'evacuated_positives']
    data = pd.DataFrame()
    for col in cols:
        assert col in raw_data.columns, f'Column "{col}" is missing from the loaded data'
    data = raw_data[cols]

    return data


