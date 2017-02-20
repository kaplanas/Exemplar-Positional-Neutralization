from wordform import WordForm
from wordform import cases
from utils import predict_lemma, cloud_form
from random import uniform, choice
from copy import deepcopy
from statistics import mean
from parameters import *

class Agent:
    """A class for simulated linguistic agents."""
    
    def __init__(self, agent_id, initial_exemplars = None):
        """Initialize with specified exemplars (WordForms)."""
        # Set the Agent's initial set of exemplars to the set provided.  If
        # nothing was provided, set it to an empty set.
        self.agent_id = agent_id
        self.exemplars = initial_exemplars
        if initial_exemplars is None:
            self.exemplars = []
    
    def add_exemplar(self, new_exemplar):
        """Add a single exemplar to the Agent's cloud."""
        # If the Agent's cloud is already at the maximum size, randomly replace
        # an existing exemplar.
        target_cloud = [e for e in self.exemplars
                        if e.lemma == new_exemplar.lemma
                           and e.case == new_exemplar.case]
        if len(target_cloud) == max_cloud_size:
            old_exemplar = choice(target_cloud)
            self.exemplars[self.exemplars.index(old_exemplar)] = new_exemplar
        # Otherwise, just add the new exemplar.
        else:
            self.exemplars.append(new_exemplar)
    
    def add_exemplars(self, new_exemplars):
        """Add several new exemplars to the Agent's cloud."""
        # Iterate through the exemplars and add them one by one.
        for e in new_exemplars:
            self.add_exemplar(e)
    
    def get_lemmas(self):
        """Return all lemmas represented in this Agent's cloud."""
        # Get the set of lemma numbers represented in the Agent's cloud.
        return {e.lemma for e in self.exemplars}
    
    def print_exemplars(self):
        """Pretty-print the Agent's exemplars."""
        col_spacer = '  '
        # Determine the length of the longest lemma number and set the spacing
        # accordingly.
        my_lemmas = self.get_lemmas()
        longest_lemma = max(len(str(l)) for l in my_lemmas)
        print('{:{width}}'.format('', width = longest_lemma), end = '')
        # Print column headers with the names of the paradigm cells.
        for case in cases:
            case_name = cases[case]['name']
            print(col_spacer + '{:{width}}'.format(case_name,
                                                   width = len(case_name)),
                  end = '')
        print()
        # Print each lemma.
        for l in my_lemmas:
            print('{:<{width}}'.format(l, width = longest_lemma), end = '')
            # Print one example in each cell of the lemma's paradigm.
            for case in cases:
                case_name = cases[case]['name']
                forms = [e for e in self.exemplars
                         if e.lemma == l and e.case == case]
                form = cloud_form(forms)
                print(col_spacer + '{:<{width}}'.format(form,
                                                        width = len(case_name)),
                      end = '')
            print()
        print()
    
    def produce(self, paradigms, bias, informativity, categorization,
                unique_base):
        """Return a WordForm based on a random WordForm in the Agent's cloud."""
        # Choose a random lemma and case to produce.
        lemma = choice(list(self.get_lemmas()))
        case = choice(list(cases.keys()))
        # Choose a random exemplar to serve as the base of production.
        cloud = [e for e in self.exemplars
                 if e.lemma == lemma and e.case == case]
        production = deepcopy(choice(cloud))
        # Apply entrenchment between the production and the Agent's cloud.
        production.entrench(self.exemplars, paradigms, informativity,
                            categorization, unique_base)
        # Add noise and bias to the production.
        production.add_bias(bias)
        production.add_noise()
        return production
    
    def categorize(self, wordform, prob_esp, categorization):
        """Return the lemma to which the WordForm most likely belongs."""
        # With the probability specified in the argument, use ESP to determine
        # the category of the WordForm provided.
        if uniform(0, 1) < prob_esp:
            return wordform.lemma
        # Otherwise, guess intelligently.
        else:
            return predict_lemma(wordform, self.exemplars,
                                 method = categorization)
    
    def store(self, wordform, prob_esp, categorization):
        """Store the WordForm in the Agent's exemplar cloud."""
        # Make a copy of the exemplar to store.
        new_ex = deepcopy(wordform)
        # Categorize it and set its lemma appropriately.
        new_ex.lemma = self.categorize(new_ex, prob_esp = prob_esp,
                                       categorization = categorization)
        # Add it to the cloud.
        self.add_exemplar(new_ex)
    
    def __repr__(self):
        return 'Agent(' + str(self.agent_id) + ', ' + repr(self.exemplars) + ')'
    
    def __str__(self):
        return str(self.agent_id) + ': {' + ', '.join(str(wf) for wf in self.exemplars) + '}'
