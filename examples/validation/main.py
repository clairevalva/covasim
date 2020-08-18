# runs model
# Since this folder is not part of the Covasim module, this 
# file takes the place of an ordinary __init__.py file.

from covasim import __version__, __versiondate__ # These are ignored by star imports
from covasim import *
from parameters_02 import *
from model import *
#%% Imports and settings
import pytest
import sciris as sc

doplot = 1

# this would be how to run the model! - but how does it assign contacts?    
def test_sim(doplot=False): # If being run via pytest, turn off

    # Settings
    seed = 1
    verbose = 1

    # Create and run the simulation
    sim = Sim()
    sim.set_seed(seed)
    sim.run(verbose=verbose)
    
    np.save("text_save.npy", sim.people.values())

    # Optionally plot
    if doplot:
        sim.plot()

    return sim

test_sim()