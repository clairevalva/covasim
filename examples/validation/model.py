# based off of the model file from the cruiseship
# hopefully configured properly?

# imports
import numba as nb
import numpy as np
import pylab as pl
import sciris as sc
import covasim as cv
import parameters_02 as cova_pars

class Person(cv.Person):
    '''
    Class for a single person — pulled directly from the cruiseship model
    '''
    def __init__(self, age=0, sex=0, crew=False, contacts=0, uid = 0):
        self.uid      = str(uid) # Unique identifier for this person
        # self.uid      = str(uid)
        # self.age      = float(age) # Age of the person (in years)
        # self.sex      = sex # Female (0) or male (1)
        # self.crew     = crew # Wehther the person is a crew member
        self.contacts = contacts # Determine how many contacts they have

        # Define state
        self.alive       = True
        self.susceptible = True
        self.exposed     = False
        self.infectious  = False
        self.diagnosed   = False
        self.recovered   = False

        # Keep track of dates
        self.date_exposed    = None
        self.date_infectious = None
        self.date_diagnosed  = None
        self.date_recovered  = None
        return
    
class Sim(cv.BaseSim):
    '''
    The Sim class handles the running of the simulation: the number of persons,
    number of time points, and the parameters of the simulation.
    '''
    
    def __init__(self, pars=None, datafile=None):
        if pars is None:
            pars = cova_pars.make_pars()
        super().__init__(pars) # Initialize and set the parameters as attributes
        self.data = None # can/should just comment this out?
        self.set_seed(self['rand_seed'])
        self.init_results()
        self.init_people()
        self.interventions = {}
        return
    
    
    def init_results(self):
        ''' Initialize results '''
        self.results_keys = [
            'n_susceptible',
            'n_exposed',
            'n_infectious',
            'n_recovered',
            'infections',
            'tests',
            'diagnoses',
            'recoveries',
            'cum_exposed',
            'cum_tested',
            'cum_diagnosed',
            'evac_diagnoses',]
        self.results = {}
        for key in self.results_keys:
            self.results[key] = np.zeros(int(self.npts))
        self.results['t'] = np.arange(int(self.npts))
        self.results['transtree'] = {} # For storing the transmission tree
        self.results['ready'] = False
        return
    
    
    def init_people(self, seed_infections=1):
        ''' Create the people '''
        self.people = sc.odict() # Dictionary for storing the people
        self.off_ship = sc.odict() # For people who've been moved off the ship
        guests = [0]*self['n_ppl']
        class_size = 20
        conned = sc.mergedicts({'s1':class_size, 's2':class_size, 's3':class_size, 'c':class_size}, contacts)
        
        # generate contacts
        contacts_all = make_hybrid_contacts(self['n_ppl'], conned)
        nu = -1
        for is_crew in guests: # Loop over each person
            nu += 1
            person = Person(contacts=contacts_all[0][nu], uid = nu) # Create the person, gives the correct contacts
            self.people[person.uid] = person # Save them to the dictionary
            
            

        # Create the seed infections
        for i in range(seed_infections):
            person = self.people[i]
            person.susceptible = False
            person.exposed = True
            person.infectious = True
            person.date_exposed = 0
            person.date_infectious = 0

        return
    
     def summary_stats(self):
        ''' Compute the summary statistics to display at the end of a run '''
        keys = ['n_susceptible', 'n_exposed', 'n_infectious']
        summary = {}
        for key in keys:
            summary[key] = self.results[key][-1]
        return summary

    
    def run(self, seed_infections=1, verbose=None, calc_likelihood=False, do_plot=False, **kwargs):
        ''' Run the simulation '''

        T = sc.tic()

        # Reset settings and results
        if verbose is None:
            verbose = self['verbose']
        self.init_results()
        self.init_people(seed_infections=seed_infections) # Actually create the people
        
        # Main simulation loop
        for t in range(self.npts):

            # Print progress
            if verbose>=1:
                string = f'  Running day {t:0.0f} of {self["n_days"]}...'
                if verbose>=2:
                    sc.heading(string)
                else:
                    print(string)

            test_probs = {} # Store the probability of each person getting tested

            # Update each person
            for person in self.people.values():

                # Count susceptibles
                if person.susceptible:
                    self.results['n_susceptible'][t] += 1
                    continue # Don't bother with the rest of the loop

                # Handle testing probability
                if person.infectious:
                    test_probs[person.uid] = self['symptomatic'] # They're infectious: high probability of testing
                else:
                    test_probs[person.uid] = 1.0

                # If exposed, check if the person becomes infectious
                if person.exposed:
                    self.results['n_exposed'][t] += 1
                    if not person.infectious and t >= person.date_infectious: # It's the day they become infectious
                        person.infectious = True
                        if verbose>=2:
                            print(f'      Person {person.uid} became infectious!')
                    if t == person.date_infectious:
                        # give 50% chance of quaratine?
                        q_per = .5
                        quar = np.random.choice([True, False], p = [q_per, 1 - q_per])
                        person.quarantine = np.copy(quar)

                # If infectious, check if anyone gets infected
                if person.infectious:
                    # First, check for recovery
                    if person.date_recovered and t >= person.date_recovered: # It's the day they become infectious
                        person.exposed = False
                        person.infectious = False
                        person.recovered = True
                        self.results['recoveries'][t] += 1
                    else:
                        self.results['n_infectious'][t] += 1 # Count this person as infectious
                        
                        n_contacts = pt(person.contacts) # Draw the number of Poisson contacts for this person
                        # this should instead initiate the random contacts? 
                        c_contacts = cv.choose(max_n=len(self.people), n=n_contacts) # Choose people at random
                        
                        contact_inds = c_contacts
                        
                        for contact_ind in contact_inds and quar == False:
                            exposure = bt(self['r_contact']) # Check for exposure per person
                            if exposure:
                                target_person = self.people[contact_ind]
                                if target_person.susceptible: # Skip people who are not susceptible
                                    self.results['infections'][t] += 1
                                    target_person.susceptible = False
                                    target_person.exposed = True
                                    target_person.date_exposed = t
                                    incub_pars = dict(dist='normal_int', par1=self['incub'], par2=self['incub_std'])
                                    dur_pars   = dict(dist='normal_int', par1=self['dur'],   par2=self['dur_std'])
                                    incub_dist = cv.sample(**incub_pars)
                                    dur_dist   = cv.sample(**dur_pars)

                                    target_person.date_infectious = t + incub_dist
                                    target_person.date_recovered = target_person.date_infectious + dur_dist
                                    if verbose>=2:
                                        print(f'        Person {person.uid} infected person {target_person.uid}!')

                # Count people who recovered
                if person.recovered:
                    self.results['n_recovered'][t] += 1

    