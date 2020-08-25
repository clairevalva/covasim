'''
Defines functions for making the population — pulled from population.py
'''

#Imports
import numpy as np  # Needed for a few things not provided by pl
import sciris as sc
from collections import defaultdict
import covasim as cova
import covasim.requirements as cvreq
import covasim.utils as cvu
import covasim.misc as cvm
import covasim.defaults as cvd
import covasim.parameters as cvpars
import covasim.people as cvppl
import random


# Specify all externally visible functions this file defines
__all__ = ['make_people', 'make_randpop', 'make_random_contacts',
           'make_microstructured_contacts', 'make_hybrid_contacts']


def make_people(sim, save_pop=False, popfile=None, die=True, reset=False, verbose=None, **kwargs):
    '''
    Make the actual people for the simulation. Usually called via sim.initialize(),
    not directly by the user.
    Args:
        sim (Sim): the simulation object
        save_pop (bool): whether to save the population to disk
        popfile (bool): if so, the filename to save to
        die (bool): whether or not to fail if synthetic populations are requested but not available
        reset (bool): whether to force population creation even if self.popdict/self.people exists
        verbose (bool): level of detail to print
        kwargs (dict): passed to make_randpop() or make_synthpop()
    Returns:
        people (People): people
    '''

    # Set inputs and defaults
    pop_size = int(sim['pop_size']) # Shorten
    pop_type = sim['pop_type'] # Shorten
    if verbose is None:
        verbose = sim['verbose']
    if popfile is None:
        popfile = sim.popfile

    # Check which type of population to produce
    if pop_type == 'synthpops':
        if not cvreq.check_synthpops():
            errormsg = f'You have requested "{pop_type}" population, but synthpops is not available; please use random, clustered, or hybrid'
            if die:
                raise ValueError(errormsg)
            else:
                print(errormsg)
                pop_type = 'random'

        location = sim['location']
        if location:
            print(f'Warning: not setting ages or contacts for "{location}" since synthpops contacts are pre-generated')

    # Actually create the population
    if sim.people and not reset:
        return sim.people # If it's already there, just return
    elif sim.popdict and not reset:
        popdict = sim.popdict # Use stored one
        sim.popdict = None # Once loaded, remove
    else:
        # Create the population
        if pop_type in ['random', 'clustered', 'uni']:
            # calls make_randpop -- should have the argument chosen here
            popdict = make_randpop(sim, microstructure=pop_type)
        elif pop_type is None:
            errormsg = f'You have set pop_type=None. This is fine, but you must ensure sim.popdict exists before calling make_people().'
            raise ValueError(errormsg)
        else:
            errormsg = f'Population type "{pop_type}" not found; choices are random, clustered, hybrid, or synthpops'
            raise ValueError(errormsg)

    # Ensure prognoses are set
    if sim['prognoses'] is None:
        sim['prognoses'] = cvpars.get_prognoses(sim['prog_by_age'])

    # Actually create the people
    people = cvppl.People(sim.pars, uid=popdict['uid'], age=popdict['age'],
                          sex=popdict['sex'], contacts=popdict['contacts']) # List for storing the people

    average_age = sum(popdict['age']/pop_size)
    sc.printv(f'Created {pop_size} people, average age {average_age:0.2f} years', 2, verbose)

    if save_pop:
        if popfile is None:
            errormsg = 'Please specify a file to save to using the popfile kwarg'
            raise FileNotFoundError(errormsg)
        else:
            filepath = sc.makefilepath(filename=popfile)
            cvm.save(filepath, people)
            if verbose:
                print(f'Saved population of type "{pop_type}" with {pop_size:n} people to {filepath}')

    return people

def choose_r(max_n, n):
    '''
    Choose a subset of items (e.g., people), with replacement.

    Args:
        max_n (int): the total number of items
        n (int): the number of items to choose

    **Example**::

        choices = cv.choose_r(5, 10) # choose 10 out of 5 people with equal probability (with repeats)
    '''
    return np.random.choice(max_n, n, replace=True)

def make_randpop(sim,  microstructure="uni"):
    '''
    Make a random population, with contacts.
    This function returns a "popdict" dictionary, which has the following (required) keys:
        - uid: an array of (usually consecutive) integers of length N, uniquely identifying each agent
        - age: an array of floats of length N, the age in years of each agent
        - sex: an array of integers of length N (not currently used, so does not have to be binary)
        - contacts: list of length N listing the contacts; see make_random_contacts() for details
        - layer_keys: a list of strings representing the different contact layers in the population; see make_random_contacts() for details
    Args:
        sim (Sim): the simulation object
        use_age_data (bool): whether to use location-specific age data
        use_household_data (bool): whether to use location-specific household size data
        sex_ratio (float): proportion of the population that is male (not currently used)
        microstructure (bool): whether or not to use the microstructuring algorithm to group contacts
    Returns:
        popdict (dict): a dictionary representing the population, with the following keys for a population of N agents with M contacts between them:
    '''

    pop_size = int(sim['pop_size']) # Number of people


    # Handle sexes and ages
    uids           = np.arange(pop_size, dtype=cvd.default_int)
    sexes          = np.random.binomial(1, 0, pop_size)
    age_data_min   = 18
    age_data_max   = 23 + 1 # changed these for purposes of the sim
    ages           = np.array([20 for _ in range(pop_size)]) # just make everyone age 20
    
    # Store output
    popdict = {}
    popdict['uid'] = uids
    popdict['age'] = ages
    popdict['sex'] = sexes

    # Actually create the contacts
    if microstructure == 'random':    contacts, layer_keys    = make_random_contacts(pop_size, sim['contacts'])
    elif microstructure == 'clustered': contacts, layer_keys, _ = make_microstructured_contacts(pop_size, sim['contacts'])
    elif microstructure == 'uni':    contacts, layer_keys, _ = make_hybrid_contacts(pop_size, ages, sim['contacts'])
    else:
        errormsg = f'Microstructure type "{microstructure}" not found; choices are random, clustered, or hybrid'
        raise NotImplementedError(errormsg)

    popdict['contacts']   = contacts
    popdict['layer_keys'] = layer_keys

    return popdict

def make_random_contacts(pop_size, contacts, overshoot=1.2, dispersion=None):
    '''
    Make random static contacts.
    Args:
        pop_size (int): number of agents to create contacts between (N)
        contacts (dict): a dictionary with one entry per layer describing the average number of contacts per person for that layer
        overshoot (float): to avoid needing to take multiple Poisson draws
        dispersion (float): if not None, use a negative binomial distribution with this dispersion parameter instead of Poisson to make the contacts
    Returns:
        contacts_list (list): a list of length N, where each entry is a dictionary by layer, and each dictionary entry is the UIDs of the agent's contacts
        layer_keys (list): a list of layer keys, which is the same as the keys of the input "contacts" dictionary
    '''

    # Preprocessing
    pop_size = int(pop_size) # Number of people
    contacts = sc.dcp(contacts)
    layer_keys = list(contacts.keys())
    contacts_list = []

    # Precalculate contacts
    n_across_layers = np.sum(list(contacts.values()))
    n_all_contacts  = int(pop_size*n_across_layers*overshoot) # The overshoot is used so we won't run out of contacts if the Poisson draws happen to be higher than the expected value
    all_contacts    = choose_r(max_n=pop_size, n=n_all_contacts) # Choose people at random
    p_counts = {}
    for lkey in layer_keys:
        if dispersion is None:
            p_count = cvu.n_poisson(contacts[lkey], pop_size) # Draw the number of Poisson contacts for this person
        else:
            p_count = cvu.n_neg_binomial(rate=contacts[lkey], dispersion=dispersion, n=pop_size) # Or, from a negative binomial
        p_counts[lkey] = np.array((p_count/2.0).round(), dtype=cvd.default_int)

    # Make contacts
    count = 0
    for p in range(pop_size):
        contact_dict = {}
        for lkey in layer_keys:
            n_contacts = p_counts[lkey][p]
            contact_dict[lkey] = all_contacts[count:count+n_contacts] # Assign people
            count += n_contacts
        contacts_list.append(contact_dict)

    return contacts_list, layer_keys
                              
def my_shuffle(array):
    random.shuffle(array)
    return array                              
                              
def make_microstructured_contacts(pop_size, contacts):
    ''' Create microstructured contacts -- i.e. for households '''

    # Preprocessing -- same as above
    pop_size = int(pop_size) # Number of people
    contacts = sc.dcp(contacts)
    contacts.pop('c', None) # Remove community
    layer_keys = list(contacts.keys())
    contacts_list = [{c:[] for c in layer_keys} for p in range(pop_size)] # Pre-populate
    
    shuffled = np.array([my_shuffle(np.array(range(pop_size))) for _ in range(3)]) # will have to change if add more layers
    
    layer_ind = 0

    for layer_name, cluster_size in contacts.items():

        # Initialize
        cluster_dict = dict() # Add dictionary for this layer
        n_remaining = pop_size # Make clusters - each person belongs to one cluster
        contacts_dict = defaultdict(set) # Use defaultdict of sets for convenience while initializing. Could probably change this as part of performance optimization

        # Loop over the clusters
        cluster_id = -1
        while n_remaining > 0:
            cluster_id += 1 # Assign cluster id
            this_cluster =  np.random.poisson(cluster_size)  # Sample the cluster size
            if this_cluster > n_remaining:
                this_cluster = n_remaining

            # Indices of people in this cluster
            cluster_indices = (pop_size-n_remaining)+np.arange(this_cluster)
            actual_add = shuffled[layer_ind][cluster_indices]
            cluster_dict[cluster_id] = actual_add#cluster_indices
            for i in actual_add: #cluster_indices: # Add symmetric pairwise contacts in each cluster
                for j in actual_add: #cluster_indices:
                    if j > i:
                        contacts_dict[i].add(j)

            n_remaining -= this_cluster
        layer_ind += 1

        for key in contacts_dict.keys():
            contacts_list[key][layer_name] = np.array(list(contacts_dict[key]), dtype=int)

        clusters = {layer_name: cluster_dict}

    return contacts_list, layer_keys, clusters




def make_hybrid_contacts(pop_size, contacts):
    '''
    Create "hybrid" contacts -- microstructured contacts for households and
    random contacts for schools and workplaces, both of which have extremely
    basic age structure. A combination of both make_random_contacts() and
    make_microstructured_contacts().
    '''

    # Handle inputs and defaults
    layer_keys = ['s1', 's2', 's3', 'c']
    contacts = sc.mergedicts({'s1':21, 's2':21, 's3':23, 'c':20}, contacts) # Ensure essential keys are populated
    

    # Create the empty contacts list -- a list of {'h':[], 's':[], 'w':[]}
    contacts_list = [{key:[] for key in layer_keys} for i in range(pop_size)]

    # make school contacts for each thing
    s1_contacts, _, clusters = make_microstructured_contacts(pop_size,
                                                             {'s1':contacts['s1']})
    s2_contacts, _, clusters = make_microstructured_contacts(pop_size,
                                                             {'s2':contacts['s2']})
     
    s3_contacts, _, clusters = make_microstructured_contacts(pop_size,
                                                             {'s3':contacts['s3']})                      

    # Make community contacts
    c_contacts, _ = make_random_contacts(pop_size, {'c':contacts['c']})

   
   
    # Construct the actual lists of contacts
    for i in range(pop_size):
        contacts_list[i]['s1'] = s1_contacts[i]['s1']  
        contacts_list[i]['s2'] = s2_contacts[i]['s2']  
        contacts_list[i]['s3'] = s3_contacts[i]['s3']  
        contacts_list[i]['c'] = c_contacts[i]['c']
    
    
    return contacts_list, layer_keys, clusters

